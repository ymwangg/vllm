import gc
import time
import warnings
from collections import defaultdict
from typing import Dict, List, NamedTuple, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from vllm.attention import AttentionMetadata, get_attn_backend
from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ParallelConfig, SchedulerConfig,
                         VisionLanguageConfig, SpeculativeConfig)
from vllm.distributed import broadcast_tensor_dict
from vllm.distributed.parallel_state import graph_capture
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sampling_params import SamplingParams
from vllm.sequence import SamplerOutput, SequenceData, SequenceGroupMetadata, SpeculateOutput, SpeculateSequenceGroupOutput
from vllm.utils import (CudaMemoryProfiler, get_kv_cache_torch_dtype, is_hip,
                        is_pin_memory_available, make_tensor_with_pad)
from vllm.distributed.communication_op import tensor_model_parallel_all_gather
from vllm.distributed.parallel_state import MarkActiveModel, ActiveModel

logger = init_logger(__name__)

_PAD_SLOT_ID = -1
LORA_WARMUP_RANK = 8
_BATCH_SIZE_ALIGNMENT = 8
# Capture graphs for token size 1, 2, 4, 8, 16, 24, 32, 40, ..., 256.
# NOTE: _get_graph_batch_size needs to be updated if this list is changed.
_BATCH_SIZES_TO_CAPTURE = [1, 2, 4] + [
    _BATCH_SIZE_ALIGNMENT * i for i in range(1, 33)
]
_NUM_WARMUP_ITERS = 2


class ModelInput(NamedTuple):
    input_tokens: torch.Tensor
    input_positions: torch.Tensor
    attn_metadata: Optional[AttentionMetadata]
    seq_lens: List[int]
    query_lens: List[int]
    lora_mapping: Optional[LoRAMapping]
    lora_requests: Set[LoRARequest]
    multi_modal_kwargs: Dict[str, torch.Tensor]
    slot_mapping: torch.Tensor
    num_prefill_tokens: int
    num_decode_tokens: int
    num_prefills: int

    @classmethod
    def empty(cls, device):
        return ModelInput(
            input_tokens=torch.empty(0, device=device),
            input_positions=torch.empty(0, device=device),
            attn_metadata=None,
            seq_lens=[],
            query_lens=[],
            lora_mapping=None,
            lora_requests=set(),
            multi_modal_kwargs={},
            slot_mapping=torch.empty(0, device=device),
            num_prefill_tokens=0,
            num_decode_tokens=0,
            num_prefills=0,
        )


class ModelRunner:

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        vision_language_config: Optional[VisionLanguageConfig] = None,
        rank: int = 0,
        speculative_config: Optional[SpeculativeConfig] = None,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        self.load_config = load_config
        self.is_driver_worker = is_driver_worker
        self.vision_language_config = vision_language_config

        self.device = self.device_config.device
        self.pin_memory = is_pin_memory_available()

        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        self.max_seq_len_to_capture = self.model_config.max_seq_len_to_capture
        self.graph_runners: Dict[int, CUDAGraphRunner] = {}
        self.graph_memory_pool: Optional[Tuple[
            int, int]] = None  # Set during graph capture.
        # When using CUDA graph, the input block tables must be padded to
        # max_seq_len_to_capture. However, creating the block table in
        # Python can be expensive. To optimize this, we cache the block table
        # in numpy and only copy the actual input content at every iteration.
        # The shape of the cached block table will be
        # (max batch size to capture, max context len to capture / block size).
        self.graph_block_tables = np.zeros(
            (max(_BATCH_SIZES_TO_CAPTURE), self.get_max_block_per_batch()),
            dtype=np.int32)
        self.attn_backend = get_attn_backend(
            self.model_config.get_num_attention_heads(self.parallel_config),
            self.model_config.get_head_size(),
            self.model_config.get_num_kv_heads(self.parallel_config),
            self.model_config.get_sliding_window(),
            self.model_config.dtype,
            self.kv_cache_dtype,
            self.block_size,
        )

        # Create processor for multi-modal data
        if self.vision_language_config is not None:
            self.multi_modal_input_processor = MULTIMODAL_REGISTRY \
                .create_input_processor(
                    self.model_config,
                    self.vision_language_config,
                )
        else:
            self.multi_modal_input_processor = None

        # Lazy initialization
        self.model: nn.Module  # Set after load_model
        # Set if the backend is flashinfer.
        self.flashinfer_workspace_buffer: torch.Tensor
        # Set after load_model.
        self.lora_manager: Optional[LRUCacheWorkerLoRAManager] = None
        # speculative decoding related variables
        self.rank = rank
        self.draft_model: torch.nn.Module  # set after load_model
        if speculative_config:
            self.use_speculate = True
            self.draft_model_config = speculative_config.draft_model_config
            self.speculate_length = speculative_config.num_speculative_tokens
        else:
            self.use_speculate = False
            self.draft_model_config = None
            self.speculate_length = None

        self.d_graph_runners: Dict[int, CUDAGraphRunner] = {}
        self.d_graph_memory_pool = None  # Set during graph capture

    def load_model(self) -> None:
        with CudaMemoryProfiler() as m:
            self.model = get_model(
                model_config=self.model_config,
                device_config=self.device_config,
                load_config=self.load_config,
                lora_config=self.lora_config,
                vision_language_config=self.vision_language_config,
                parallel_config=self.parallel_config,
                scheduler_config=self.scheduler_config,
                cache_config=self.cache_config,
            )

        self.model_memory_usage = m.consumed_memory
        logger.info("Loading model weights took %.4f GB",
                    self.model_memory_usage / float(2**30))

        if self.lora_config:
            assert hasattr(self.model, "supported_lora_modules"
                           ) and self.model.supported_lora_modules, (
                               "Model does not support LoRA")
            assert hasattr(
                self.model,
                "embedding_modules"), "Model does not have embedding_modules"
            assert hasattr(self.model, "embedding_padding_modules"
                           ), "Model does not have embedding_padding_modules"
            self.lora_manager = LRUCacheWorkerLoRAManager(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens,
                self.vocab_size,
                self.lora_config,
                self.device,
                self.model.embedding_modules,
                self.model.embedding_padding_modules,
                max_position_embeddings=self.model.config.
                max_position_embeddings,
            )
            self.model = self.lora_manager.create_lora_manager(self.model)

        if self.kv_cache_dtype == "fp8" and is_hip():
            # Currently only ROCm accepts kv-cache scaling factors
            # via quantization_param_path and this will be deprecated
            # in the future.
            if self.model_config.quantization_param_path is not None:
                if callable(getattr(self.model, "load_kv_cache_scales", None)):
                    warnings.warn(
                        "Loading kv cache scaling factor from JSON is "
                        "deprecated and will be removed. Please include "
                        "kv cache scaling factors in the model checkpoint.",
                        FutureWarning,
                        stacklevel=2)
                    self.model.load_kv_cache_scales(
                        self.model_config.quantization_param_path)
                    logger.info("Loaded KV cache scaling factors from %s",
                                self.model_config.quantization_param_path)
                else:
                    raise RuntimeError(
                        "Using FP8 KV cache and scaling factors provided but "
                        "model %s does not support loading scaling factors.",
                        self.model.__class__)
            else:
                logger.warning(
                    "Using FP8 KV cache but no scaling factors "
                    "provided. Defaulting to scaling factors of 1.0. "
                    "This may lead to less accurate results!")

        # Load draft model when enabling speculative decoding
        if self.use_speculate:
            with MarkActiveModel(ActiveModel.DRAFT):
                # NOTE: We use global variable to control parallel state (world size, rank)
                # of draft model used in speculative decoding.
                with CudaMemoryProfiler() as m:
                    self.draft_model = get_model(
                        model_config=self.draft_model_config,
                        device_config=self.device_config,
                        load_config=self.load_config,
                        lora_config=self.lora_config,
                        vision_language_config=self.vision_language_config,
                        parallel_config=self.parallel_config,
                        scheduler_config=self.scheduler_config,
                    )
                self.draft_model_memory_usage = m.consumed_memory
                logger.info(
                    f"Loading draft model weights took "
                    f"{self.draft_model_memory_usage / float(2**30):.4f} GB")

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        from vllm.model_executor.model_loader.loader import ShardedStateLoader
        ShardedStateLoader.save_model(
            self.model,
            path,
            pattern=pattern,
            max_size=max_size,
        )

    def save_tensorized_model(
        self,
        tensorizer_config: TensorizerConfig,
    ) -> None:
        from vllm.model_executor.model_loader.loader import TensorizerLoader
        TensorizerLoader.save_model(
            self.model,
            tensorizer_config=tensorizer_config,
        )

    def get_max_block_per_batch(self) -> int:
        block_size = self.block_size
        return (self.max_seq_len_to_capture + block_size - 1) // block_size

    def _prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> ModelInput:
        """Prepare the model input based on a given sequence group.

        The API assumes seq_group_metadata_list is sorted by prefill -> decode.

        The result tensors and data structure also batches input in prefill
        -> decode order. For example,

        - input_tokens[:num_prefill_tokens] contains prefill tokens.
        - input_tokens[num_prefill_tokens:] contains decode tokens.

        If cuda graph is required, this API automatically pads inputs.
        """
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []
        lora_index_mapping: List[int] = []
        lora_prompt_mapping: List[int] = []
        lora_requests: Set[LoRARequest] = set()

        seq_lens: List[int] = []
        prefill_seq_lens: List[int] = []
        decode_seq_lens: List[int] = []
        context_lens: List[int] = []
        query_lens: List[int] = []
        block_tables: List[List[int]] = []
        multi_modal_kwargs_list: Dict[str,
                                      List[torch.Tensor]] = defaultdict(list)
        decode_only = True
        num_prefills = 0
        num_prefill_tokens = 0
        num_decode_tokens = 0

        # The following fields are only for flashinfer
        # Please follow https://docs.flashinfer.ai/tutorials/kv_layout.html#page-layout
        # for the precise definition of the following fields.
        # An example:
        # request 1, page indices [0, 5, 8]
        # request 2, page indices [1, 6, 7]
        # request 3, page indices [3, 4]
        # paged_kv_indices is a concatenation of page indices of all requests:
        # [0, 5, 8, 1, 6, 7, 3, 4]
        # paged_kv_indptr is used to index into paged_kv_indices:
        # [0, 3, 6, 8]
        paged_kv_indices: List[int] = []
        # 0 at the beginning of paged_kv_indptr indicates the start of the
        # first request’s page indices in the paged_kv_indices list.
        paged_kv_indptr: List[int] = [0]
        # paged_kv_last_page_len is the length of the last page of each request
        paged_kv_last_page_len: List[int] = []

        if len(seq_group_metadata_list) == 0:
            return ModelInput.empty(self.device)

        if self.sliding_window is not None:
            sliding_window_blocks = (self.sliding_window + self.block_size -
                                     1) // self.block_size
            block_aligned_sliding_window = \
                sliding_window_blocks * self.block_size

        for seq_group_metadata in seq_group_metadata_list:
            seq_ids = list(seq_group_metadata.seq_data.keys())
            is_prompt = seq_group_metadata.is_prompt

            for seq_id in seq_ids:
                computed_block_nums = seq_group_metadata.computed_block_nums
                if (self.scheduler_config is not None
                        and self.scheduler_config.chunked_prefill_enabled
                        and not (computed_block_nums is None
                                 or computed_block_nums == [])):
                    raise RuntimeError(
                        "chunked prefill cannot be used with prefix caching "
                        "now.")

                seq_data = seq_group_metadata.seq_data[seq_id]
                if is_prompt:
                    context_len = seq_data.get_num_computed_tokens()
                else:
                    # get_num_computed_tokens is incorrect for spec decoding.
                    # So, we should have a special logic here.
                    # TODO(sang): Fix it.
                    context_len = seq_data.get_len() - 1

                seq_len = min(
                    seq_data.get_len(),
                    context_len + seq_group_metadata.token_chunk_size)
                if is_prompt:
                    tokens = seq_data.get_token_ids()[context_len:seq_len]
                else:
                    # Optimization. get_token_ids requires the entire copy of
                    # tokens.
                    tokens = [seq_data.get_last_token_id()]

                # Prefix cache was hit.
                # Prefix is not supported with sliding_window
                prefix_cache_hit = (computed_block_nums is not None
                                    and len(computed_block_nums) > 0
                                    and self.sliding_window is None
                                    and is_prompt)

                # These are seq_len/context_len capped to the sliding window.
                # They are passed to decode kernel.
                # We still need original seq_len/context_len to compute slot
                # mapping (and input position) below.
                curr_sliding_window_blocks = None
                sliding_seq_len = seq_len
                sliding_context_len = context_len

                # TODO(sang): This is a hack to make sliding window work with
                # paged attn. We can remove it if we make paged attn kernel
                # to properly handle slinding window attn.
                if (self.sliding_window is not None and not is_prompt):
                    curr_sliding_window_blocks = sliding_window_blocks
                    if self.scheduler_config.use_v2_block_manager:
                        # number of elements in last block
                        suff_len = seq_len % self.block_size
                        sliding_seq_len = min(
                            seq_len, block_aligned_sliding_window + suff_len)
                        if suff_len > 0:
                            curr_sliding_window_blocks += 1
                    else:
                        sliding_seq_len = min(seq_len, self.sliding_window)
                    sliding_context_len = sliding_seq_len - 1

                # TODO(sang): Combine chunked prefill and prefix caching by
                # only allowing multiple of block_size chunk size.
                # NOTE: This only works for oooooooxxx style attention.
                if prefix_cache_hit:
                    assert computed_block_nums is not None
                    context_len = len(computed_block_nums) * self.block_size
                    tokens = tokens[context_len:]

                    # need to think what to set it to when we have both sliding
                    # window and prefix caching...
                    assert self.sliding_window is None, \
                        "Prefix caching is not supported with sliding window"
                    sliding_context_len = context_len

                    if self.attn_backend.get_name() == "flash-attn":
                        # NOTE(woosuk): For flash-attn, the block table should
                        # include the entries for the incoming prefill tokens.
                        # TODO(woosuk): This is a temporary fix. We should
                        # provide a unified interface for different backends.
                        block_table = seq_group_metadata.block_tables[seq_id]
                    else:
                        block_table = computed_block_nums
                elif (self.scheduler_config.chunked_prefill_enabled
                      or not is_prompt):
                    if seq_group_metadata.block_tables is not None:
                        # chunked prefill or decode
                        block_table = seq_group_metadata.block_tables[seq_id]
                        if curr_sliding_window_blocks is not None:
                            block_table = block_table[
                                -curr_sliding_window_blocks:]
                        if self.attn_backend.get_name() == "flashinfer":
                            paged_kv_indices.extend(block_table)
                            paged_kv_indptr.append(paged_kv_indptr[-1] +
                                                   len(block_table))
                            last_page_len = seq_data.get_len(
                            ) % self.block_size
                            if last_page_len == 0:
                                last_page_len = self.block_size
                            paged_kv_last_page_len.append(last_page_len)
                    else:
                        # Only happens when memory profiling runs.
                        block_table = []
                else:
                    # Prefill without chunked prefill or memory profiling.
                    block_table = []
                block_tables.append(block_table)

                seq_lens.append(sliding_seq_len)
                context_lens.append(sliding_context_len)
                query_len = sliding_seq_len - sliding_context_len
                query_lens.append(query_len)
                input_tokens.extend(tokens)
                input_positions.extend(list(range(context_len, seq_len)))
                lora_id = seq_group_metadata.lora_int_id

                if is_prompt:
                    assert len(seq_ids) == 1
                    num_prefills += 1
                    num_prefill_tokens += len(tokens)
                    decode_only = False
                    prefill_seq_lens.append(seq_len)
                else:
                    assert query_len == 1, (
                        "seq_len: {}, context_len: {}, query_len: {}".format(
                            seq_len, context_len, query_len))
                    num_decode_tokens += query_len
                    decode_seq_lens.append(sliding_seq_len)

                if lora_id > 0:
                    lora_requests.add(seq_group_metadata.lora_request)

                lora_index_mapping += [lora_id] * query_len
                lora_prompt_mapping.extend(
                    [lora_id] *
                    (query_len if seq_group_metadata.sampling_params
                     and seq_group_metadata.sampling_params.prompt_logprobs
                     is not None else 1))

                mm_data = seq_group_metadata.multi_modal_data
                if mm_data is not None:
                    # Process multi-modal data
                    if self.multi_modal_input_processor is None:
                        raise ValueError(
                            "Multi-modal inputs are only supported by "
                            "vision language models.")

                    mm_kwargs = self.multi_modal_input_processor(mm_data)
                    for k, v in mm_kwargs.items():
                        multi_modal_kwargs_list[k].append(v)

                if _is_block_tables_empty(seq_group_metadata.block_tables):
                    # During memory profiling, the block tables are not
                    # initialized yet. In this case, we just use a dummy
                    # slot mapping.
                    # In embeddings, the block tables are {seq_id: None}.
                    slot_mapping.extend([_PAD_SLOT_ID] * seq_len)
                    continue

                # Compute the slot mapping.
                block_table = seq_group_metadata.block_tables[seq_id]

                # Mask the [0, start_idx) tokens of the prompt with
                # _PAD_SLOT_ID, where start_idx is max(0, seq_len -
                # sliding_window). For example, if the prompt len is 10,
                # sliding window is 8, and block size is 4, the first two
                # tokens are masked and the slot mapping will be
                # [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
                start_idx = 0
                if self.sliding_window is not None:
                    if is_prompt:
                        assert self.scheduler_config.use_v2_block_manager \
                            or context_len == 0, (
                            "Prefix caching is currently not supported with "
                            "sliding window attention in V1 block manager")
                    # It is an optimization. When it is decoding, it is always
                    # 0. When prefill, we use it to not write slots to kv cache
                    # to save memory.
                    start_idx = max(0, query_len - self.sliding_window)

                for i in range(context_len, seq_len):
                    if i < start_idx:
                        slot_mapping.append(_PAD_SLOT_ID)
                        continue

                    block_number = block_table[i // self.block_size]
                    block_offset = i % self.block_size
                    slot = block_number * self.block_size + block_offset
                    slot_mapping.append(slot)

        batch_size = len(input_tokens)
        max_query_len = max(query_lens)
        max_prefill_seq_len = max(prefill_seq_lens, default=0)
        max_decode_seq_len = max(decode_seq_lens, default=0)

        # If cuda graph can be used, pad tensors accordingly.
        # See `capture_model` API for more details.
        # vLLM uses cuda graph only for decoding requests.
        use_captured_graph = (
            decode_only and not self.model_config.enforce_eager
            and batch_size <= _BATCH_SIZES_TO_CAPTURE[-1]
            and max_decode_seq_len <= self.max_seq_len_to_capture)
        if use_captured_graph:
            graph_batch_size = _get_graph_batch_size(batch_size)
            assert graph_batch_size >= batch_size
            for _ in range(graph_batch_size - batch_size):
                input_tokens.append(0)
                input_positions.append(0)
                slot_mapping.append(_PAD_SLOT_ID)
                seq_lens.append(1)
                block_tables.append([])
                lora_index_mapping.append(0)
            batch_size = graph_batch_size
            num_decode_tokens = batch_size

        if use_captured_graph:
            # The shape of graph_block_tables is
            # [max batch size, max context len // block size].
            input_block_tables = self.graph_block_tables[:batch_size]
            for i, block_table in enumerate(block_tables):
                if block_table:
                    input_block_tables[i, :len(block_table)] = block_table
            block_tables = torch.tensor(input_block_tables, device=self.device)
        else:
            max_block_table_len = max(
                len(block_table) for block_table in block_tables)
            block_tables = make_tensor_with_pad(
                block_tables,
                max_len=max_block_table_len,
                pad=0,
                dtype=torch.int,
                device=self.device,
            )
        assert max_query_len > 0, ("query_lens: {}".format(query_lens))

        seq_lens_tensor = torch.tensor(seq_lens,
                                       dtype=torch.int,
                                       device=self.device)
        seq_start_loc = torch.zeros(seq_lens_tensor.shape[0] + 1,
                                    dtype=torch.int32,
                                    device=self.device)

        torch.cumsum(seq_lens_tensor,
                     dim=0,
                     dtype=seq_start_loc.dtype,
                     out=seq_start_loc[1:])

        input_tokens_tensor = torch.tensor(input_tokens,
                                           dtype=torch.long,
                                           device=self.device)
        input_positions_tensor = torch.tensor(input_positions,
                                              dtype=torch.long,
                                              device=self.device)
        slot_mapping_tensor = torch.tensor(slot_mapping,
                                           dtype=torch.long,
                                           device=self.device)

        if self.attn_backend.get_name() == "flashinfer":
            if not hasattr(self, "flashinfer_workspace_buffer"):
                # Allocate 16MB workspace buffer
                # Follow the example of flashinfer: https://docs.flashinfer.ai/api/python/decode.html
                self.flashinfer_workspace_buffer = torch.empty(
                    16 * 1024 * 1024, dtype=torch.uint8, device=self.device)
            paged_kv_indptr_tensor = torch.tensor(paged_kv_indptr,
                                                  dtype=torch.int,
                                                  device=self.device)
            paged_kv_indices_tensor = torch.tensor(paged_kv_indices,
                                                   dtype=torch.int,
                                                   device=self.device)
            paged_kv_last_page_len_tensor = torch.tensor(
                paged_kv_last_page_len, dtype=torch.int, device=self.device)
            kv_cache_dtype = get_kv_cache_torch_dtype(self.kv_cache_dtype,
                                                      self.model_config.dtype)
            attn_metadata = self.attn_backend.make_metadata(
                num_prefills=num_prefills,
                slot_mapping=slot_mapping_tensor,
                num_prefill_tokens=num_prefill_tokens,
                num_decode_tokens=num_decode_tokens,
                use_cuda_graph=False,
                max_prefill_seq_len=max_prefill_seq_len,
                block_tables=block_tables,
                workspace_buffer=self.flashinfer_workspace_buffer,
                paged_kv_indptr=paged_kv_indptr_tensor,
                paged_kv_indices=paged_kv_indices_tensor,
                paged_kv_last_page_len=paged_kv_last_page_len_tensor,
                num_qo_heads=self.model_config.get_num_attention_heads(
                    self.parallel_config),
                num_kv_heads=self.model_config.get_num_kv_heads(
                    self.parallel_config),
                head_dim=self.model_config.get_head_size(),
                page_size=16,
                seq_start_loc=seq_start_loc,
                data_type=kv_cache_dtype)
        else:
            context_lens_tensor = torch.tensor(context_lens,
                                               dtype=torch.int,
                                               device=self.device)
            query_lens_tensor = torch.tensor(query_lens,
                                             dtype=torch.long,
                                             device=self.device)
            query_start_loc = torch.zeros(query_lens_tensor.shape[0] + 1,
                                          dtype=torch.int32,
                                          device=self.device)

            torch.cumsum(query_lens_tensor,
                         dim=0,
                         dtype=query_start_loc.dtype,
                         out=query_start_loc[1:])

            attn_metadata = self.attn_backend.make_metadata(
                num_prefills=num_prefills,
                slot_mapping=slot_mapping_tensor,
                num_prefill_tokens=num_prefill_tokens,
                num_decode_tokens=num_decode_tokens,
                seq_lens=seq_lens,
                seq_lens_tensor=seq_lens_tensor,
                max_query_len=max_query_len,
                max_prefill_seq_len=max_prefill_seq_len,
                max_decode_seq_len=max_decode_seq_len,
                query_start_loc=query_start_loc,
                seq_start_loc=seq_start_loc,
                context_lens_tensor=context_lens_tensor,
                block_tables=block_tables,
                use_cuda_graph=use_captured_graph,
            )

        if self.lora_config:
            lora_mapping = LoRAMapping(
                lora_index_mapping,
                lora_prompt_mapping,
            )
        else:
            lora_mapping = None

        multi_modal_kwargs = {
            k: torch.cat(v, dim=0).to(self.device)
            for k, v in multi_modal_kwargs_list.items()
        }

        return ModelInput(
            input_tokens=input_tokens_tensor,
            input_positions=input_positions_tensor,
            attn_metadata=attn_metadata,
            seq_lens=seq_lens,
            query_lens=query_lens,
            lora_mapping=lora_mapping,
            lora_requests=lora_requests,
            multi_modal_kwargs=multi_modal_kwargs,
            slot_mapping=slot_mapping_tensor,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
        )

    def prepare_input_tensors(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
    ) -> Tuple[torch.Tensor, torch.Tensor, AttentionMetadata, SamplingMetadata,
               Set[LoRARequest], LoRAMapping, Dict[str, torch.Tensor]]:
        if self.is_driver_worker:
            assert seq_group_metadata_list is not None
            # Prepare input tensors.
            (
                input_tokens,
                input_positions,
                attn_metadata,
                seq_lens,
                query_lens,
                lora_mapping,
                lora_requests,
                multi_modal_kwargs,
                slot_mapping,
                num_prefill_tokens,
                num_decode_tokens,
                num_prefills,
            ) = self._prepare_model_input(seq_group_metadata_list)
            sampling_metadata = SamplingMetadata.prepare(
                seq_group_metadata_list, seq_lens, query_lens, self.device,
                self.pin_memory)

            metadata_dict = {
                "input_tokens": input_tokens,
                "input_positions": input_positions,
                "selected_token_indices":
                sampling_metadata.selected_token_indices,
                "lora_requests": lora_requests,
                "lora_mapping": lora_mapping,
                "multi_modal_kwargs": multi_modal_kwargs,
                "num_prefill_tokens": num_prefill_tokens,
                "num_decode_tokens": num_decode_tokens,
                "slot_mapping": slot_mapping,
                "num_prefills": num_prefills,
            }
            if attn_metadata:
                metadata_dict.update(attn_metadata.asdict_zerocopy())
            broadcast_tensor_dict(metadata_dict, src=0)
        else:
            metadata_dict = broadcast_tensor_dict(src=0)
            input_tokens = metadata_dict.pop("input_tokens")
            input_positions = metadata_dict.pop("input_positions")
            selected_token_indices = metadata_dict.pop(
                "selected_token_indices")
            lora_mapping = metadata_dict.pop("lora_mapping")
            lora_requests = metadata_dict.pop("lora_requests")
            multi_modal_kwargs = metadata_dict.pop("multi_modal_kwargs")
            if metadata_dict:
                attn_metadata = self.attn_backend.make_metadata(
                    **metadata_dict)
            else:
                attn_metadata = None
            sampling_metadata = SamplingMetadata(
                seq_groups=None,
                selected_token_indices=selected_token_indices,
                categorized_sample_indices=None,
                num_prompts=0,
            )

        return (input_tokens, input_positions, attn_metadata,
                sampling_metadata, lora_requests, lora_mapping,
                multi_modal_kwargs)

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        kv_caches: List[torch.Tensor],
    ) -> Optional[SamplerOutput]:
        (input_tokens, input_positions, attn_metadata, sampling_metadata,
         lora_requests, lora_mapping, multi_modal_kwargs
         ) = self.prepare_input_tensors(seq_group_metadata_list)

        if self.lora_config:
            self.set_active_loras(lora_requests, lora_mapping)

        # Currently cuda graph is only supported by the decode phase.
        prefill_meta = attn_metadata.prefill_metadata
        decode_meta = attn_metadata.decode_metadata
        if prefill_meta is None and decode_meta.use_cuda_graph:
            graph_batch_size = input_tokens.shape[0]
            model_executable = self.graph_runners[graph_batch_size]
        else:
            model_executable = self.model

        hidden_states = model_executable(
            input_ids=input_tokens,
            positions=input_positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            **multi_modal_kwargs,
        )

        # Compute the logits.
        logits = self.model.compute_logits(hidden_states, sampling_metadata)

        # Only perform sampling in the driver worker.
        if not self.is_driver_worker:
            return None

        # Sample the next token.
        output = self.model.sample(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )

        return output

    @torch.inference_mode()
    def profile_run(self) -> None:
        # Enable top-k sampling to reflect the accurate memory usage.
        sampling_params = SamplingParams(top_p=0.99, top_k=self.vocab_size - 1)
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs
        # This represents the maximum number of different requests
        # that will have unique loras, an therefore the max amount of memory
        # consumption create dummy lora request copies from the lora request
        # passed in, which contains a lora from the lora warmup path.
        dummy_lora_requests = []
        dummy_lora_requests_per_seq = []
        if self.lora_config:
            assert self.lora_manager is not None
            with self.lora_manager.dummy_lora_cache():
                for idx in range(self.lora_config.max_loras):
                    lora_id = idx + 1
                    dummy_lora_request = LoRARequest(
                        lora_name=f"warmup_{lora_id}",
                        lora_int_id=lora_id,
                        lora_local_path="/not/a/real/path",
                    )
                    self.lora_manager.add_dummy_lora(dummy_lora_request,
                                                     rank=LORA_WARMUP_RANK)
                    dummy_lora_requests.append(dummy_lora_request)
                dummy_lora_requests_per_seq = [
                    dummy_lora_requests[idx % len(dummy_lora_requests)]
                    for idx in range(max_num_seqs)
                ]

        # Profile memory usage with max_num_sequences sequences and the total
        # number of tokens equal to max_num_batched_tokens.
        seqs: List[SequenceGroupMetadata] = []
        # Additional GPU memory may be needed for vision encoding, which needs
        # to be accounted for when calculating the GPU blocks for
        # vLLM blocker manager.
        # To exercise the worst scenario for GPU memory consumption,
        # the number of seqs (batch_size) is chosen to maximize the number
        # of images processed.
        model_config = self.model_config
        vlm_config = self.vision_language_config

        if vlm_config:
            max_num_seqs = min(
                max_num_seqs,
                int(max_num_batched_tokens / vlm_config.image_feature_size))
        for group_id in range(max_num_seqs):
            seq_len = (max_num_batched_tokens // max_num_seqs +
                       (group_id < max_num_batched_tokens % max_num_seqs))

            if vlm_config is None:
                seq_data = SequenceData([0] * seq_len)
                dummy_multi_modal_data = None
            else:
                seq_data, dummy_multi_modal_data = MULTIMODAL_REGISTRY \
                    .dummy_data_for_profiling(seq_len, model_config, vlm_config)

            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: seq_data},
                sampling_params=sampling_params,
                block_tables=None,
                lora_request=dummy_lora_requests_per_seq[group_id]
                if dummy_lora_requests_per_seq else None,
                multi_modal_data=dummy_multi_modal_data,
            )
            seqs.append(seq)

        # Run the model with the dummy inputs.
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        kv_caches = [None] * num_layers
        self.execute_model(seqs, kv_caches)
        torch.cuda.synchronize()
        return

    def remove_all_loras(self):
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        self.lora_manager.remove_all_loras()

    def set_active_loras(self, lora_requests: Set[LoRARequest],
                         lora_mapping: LoRAMapping) -> None:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        self.lora_manager.set_active_loras(lora_requests, lora_mapping)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.remove_lora(lora_id)

    def list_loras(self) -> Set[int]:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.list_loras()

    @torch.inference_mode()
    def capture_model(self, kv_caches: List[torch.Tensor]) -> None:
        """Cuda graph capture a model.

        Note that CUDA graph's performance gain is negligible if number
        of batched tokens are larger than 200. And since CUDA graph
        requires fixed sized tensors, supporting large/variable batch
        size requires high GPU memory overhead. Thus, vLLM only captures
        decoding requests. Mixed batch (chunked prefill + decoding) or
        prefill requests are not captured.

        Since it is used for decoding-only, it assumes there's only 1 token
        per sequence in the batch.
        """
        assert not self.model_config.enforce_eager
        logger.info("Capturing the model for CUDA graphs. This may lead to "
                    "unexpected consequences if the model is not static. To "
                    "run the model in eager mode, set 'enforce_eager=True' or "
                    "use '--enforce-eager' in the CLI.")
        logger.info("CUDA graphs can take additional 1~3 GiB memory per GPU. "
                    "If you are running out of memory, consider decreasing "
                    "`gpu_memory_utilization` or enforcing eager mode. "
                    "You can also reduce the `max_num_seqs` as needed "
                    "to decrease memory usage.")
        start_time = time.perf_counter()

        # Prepare dummy inputs. These will be reused for all batch sizes.
        max_batch_size = max(_BATCH_SIZES_TO_CAPTURE)
        input_tokens = torch.zeros(max_batch_size, dtype=torch.long).cuda()
        input_positions = torch.zeros(max_batch_size, dtype=torch.long).cuda()
        slot_mapping = torch.empty(max_batch_size, dtype=torch.long).cuda()
        slot_mapping.fill_(_PAD_SLOT_ID)
        seq_lens = torch.ones(max_batch_size, dtype=torch.int32).cuda()
        block_tables = torch.from_numpy(self.graph_block_tables).cuda()

        # Prepare buffer for outputs. These will be reused for all batch sizes.
        # It will be filled after the first graph capture.
        hidden_states: Optional[torch.Tensor] = None

        graph_batch_size = _get_graph_batch_size(
            self.scheduler_config.max_num_seqs)
        batch_size_capture_list = [
            bs for bs in _BATCH_SIZES_TO_CAPTURE if bs <= graph_batch_size
        ]

        with graph_capture() as graph_capture_context:
            # NOTE: Capturing the largest batch size first may help reduce the
            # memory usage of CUDA graph.
            for batch_size in reversed(batch_size_capture_list):
                # Create dummy attn_metadata.
                attn_metadata = self.attn_backend.make_metadata(
                    num_prefills=0,
                    num_prefill_tokens=0,
                    num_decode_tokens=batch_size,
                    slot_mapping=slot_mapping[:batch_size],
                    seq_lens=None,
                    seq_lens_tensor=seq_lens[:batch_size],
                    max_query_len=None,
                    max_prefill_seq_len=0,
                    max_decode_seq_len=self.max_seq_len_to_capture,
                    query_start_loc=None,
                    seq_start_loc=None,
                    context_lens_tensor=None,
                    block_tables=block_tables[:batch_size],
                    use_cuda_graph=True,
                )

                if self.lora_config:
                    lora_mapping = LoRAMapping(
                        [0] * batch_size,
                        [0] * batch_size,
                    )
                    self.set_active_loras(set(), lora_mapping)

                graph_runner = CUDAGraphRunner(self.model)
                hidden_states = graph_runner.capture(
                    input_tokens[:batch_size],
                    input_positions[:batch_size],
                    hidden_states[:batch_size]
                    if hidden_states is not None else None,
                    kv_caches,
                    attn_metadata,
                    memory_pool=self.graph_memory_pool,
                    stream=graph_capture_context.stream,
                )
                self.graph_memory_pool = graph_runner.graph.pool()
                self.graph_runners[batch_size] = graph_runner

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        # This usually takes < 10 seconds.
        logger.info("Graph capturing finished in %.0f secs.", elapsed_time)

    @torch.inference_mode()
    def speculate_capture_model(self, kv_caches: List[torch.Tensor],
                                d_kv_caches: List[torch.Tensor]) -> None:
        """Cuda graph capture a model.

        Note that CUDA graph's performance gain is negligible if number
        of batched tokens are larger than 200. And since CUDA graph
        requires fixed sized tensors, supporting large/variable batch
        size requires high GPU memory overhead. Thus, vLLM only captures
        decoding requests. Mixed batch (chunked prefill + decoding) or
        prefill requests are not captured.

        Since it is used for decoding-only, it assumes there's only 1 token
        per sequence in the batch.
        """
        assert not self.model_config.enforce_eager
        logger.info("Capturing the model for CUDA graphs. This may lead to "
                    "unexpected consequences if the model is not static. To "
                    "run the model in eager mode, set 'enforce_eager=True' or "
                    "use '--enforce-eager' in the CLI.")
        logger.info("CUDA graphs can take additional 1~3 GiB memory per GPU. "
                    "If you are running out of memory, consider decreasing "
                    "`gpu_memory_utilization` or enforcing eager mode. "
                    "You can also reduce the `max_num_seqs` as needed "
                    "to decrease memory usage.")
        start_time = time.perf_counter()

        max_batch_size = max(_BATCH_SIZES_TO_CAPTURE)
        graph_batch_size = _get_graph_batch_size(
            self.scheduler_config.max_num_seqs)
        batch_size_capture_list = [
            bs for bs in _BATCH_SIZES_TO_CAPTURE if bs <= graph_batch_size
        ]

        def capture_graph_inner(is_draft_model: bool = False,
                                caches: List[torch.Tensor] = None) -> None:
            if is_draft_model:
                seq_len = 1
                is_multi_query_mode = False
                active_model = ActiveModel.DRAFT
                graph_runners = self.d_graph_runners
                graph_memory_pool = self.d_graph_memory_pool
            else:
                seq_len = self.speculate_length + 1
                is_multi_query_mode = True
                active_model = ActiveModel.TARGET
                graph_runners = self.graph_runners
                graph_memory_pool = self.graph_memory_pool

            # Prepare dummy inputs. These will be reused for all batch sizes.
            input_tokens = torch.zeros(max_batch_size,
                                       seq_len,
                                       dtype=torch.long).cuda()
            input_positions = torch.zeros(max_batch_size,
                                          seq_len,
                                          dtype=torch.long).cuda()
            slot_mapping = torch.empty(max_batch_size,
                                       seq_len,
                                       dtype=torch.long).cuda()
            slot_mapping.fill_(_PAD_SLOT_ID)
            context_lens = torch.ones(max_batch_size, dtype=torch.int32).cuda()
            block_tables = torch.from_numpy(self.graph_block_tables).cuda()

            # NOTE(woosuk): There are 3 backends for all-reduce: custom all-reduce
            # kernel, CuPy NCCL, and PyTorch NCCL. When using CUDA graph, we use
            # either custom all-reduce kernel or CuPy NCCL. When not using CUDA
            # graph, we use either custom all-reduce kernel or PyTorch NCCL.
            # We always prioritize using custom all-reduce kernel but fall back
            # to PyTorch or CuPy NCCL if it is disabled or not supported.
            with graph_capture() as graph_capture_context:
                # NOTE: Capturing the largest batch size first may help reduce the
                # memory usage of CUDA graph.
                for batch_size in reversed(batch_size_capture_list):
                    # Create dummy attn_metadata.
                    decode_metadata = self.attn_backend.make_metadata(
                        is_prompt=False,
                        seq_lens=None,
                        seq_lens_tensor=context_lens[:batch_size],
                        max_query_len=None,
                        max_seq_len=None,
                        subquery_start_loc=None,
                        seq_start_loc=None,
                        context_lens_tensor=None,
                        block_tables=block_tables[:batch_size],
                        use_cuda_graph=True,
                        real_batch_size=batch_size,
                        seq_len=input_tokens.shape[1],
                        is_multi_query_mode=is_multi_query_mode,
                    )
                    attn_metadata = AttentionMetadata(
                        num_prefills=0,
                        num_prefill_tokens=0,
                        num_decode_tokens=batch_size,
                        # NOTE: slot_mapping is 2d, we need to flatten it to
                        # 1d after slicing.
                        slot_mapping=slot_mapping[:batch_size].view(-1),
                        prefill_metadata=None,
                        decode_metadata=decode_metadata,
                        kv_cache_dtype=self.kv_cache_dtype,
                    )
                    graph_runner = CUDAGraphRunner(self.draft_model) if is_draft_model \
                        else CUDAGraphRunner(self.model)
                    with MarkActiveModel(active_model):
                        graph_runner.capture(
                            # NOTE: input_tokens and input_positions are 2d, we need to
                            # flatten them to 1d after slicing.
                            input_tokens[:batch_size].view(-1),
                            input_positions[:batch_size].view(-1),
                            caches,
                            attn_metadata,
                            memory_pool=graph_memory_pool,
                        )
                    graph_memory_pool = graph_runner.graph.pool()
                    graph_runners[batch_size] = graph_runner
                if is_draft_model:
                    self.d_graph_memory_pool = graph_memory_pool
                else:
                    self.graph_memory_pool = graph_memory_pool

        # 1. Capture draft model
        capture_graph_inner(is_draft_model=True, caches=d_kv_caches)
        # 2. Capture target model
        capture_graph_inner(is_draft_model=False, caches=kv_caches)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        # This usually takes < 10 seconds.
        logger.info(f"Graph capturing finished in {elapsed_time:.0f} secs.")

    def _prepare_speculate_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        is_multi_query_mode: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, AttentionMetadata,
               SamplingMetadata]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        slot_mapping: List[List[int]] = []
        context_lens: List[int] = []
        block_tables: List[List[int]] = []
        seq_groups: List[Tuple[List[int], SamplingParams]] = []
        seq_data_dict: Dict[int, SequenceData] = {}

        for seq_group_metadata in seq_group_metadata_list:
            assert not seq_group_metadata.is_prompt
            assert seq_group_metadata.token_chunk_size == 1
            seq_data_dict.update(seq_group_metadata.seq_data)

            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]

                num_generation_tokens = self.speculate_length + 1
                generation_tokens = [seq_data.get_last_token_id()
                                     ] + [0] * self.speculate_length

                input_tokens.append(generation_tokens)
                seq_len = seq_data.get_len() + self.speculate_length

                context_len = seq_len if self.sliding_window is None else min(
                    seq_len, self.sliding_window)
                context_lens.append(context_len)

                first_position = seq_len - num_generation_tokens
                positions = [
                    first_position + offset
                    for offset in range(num_generation_tokens)
                ]
                input_positions.append(positions)

                block_table = seq_group_metadata.block_tables[seq_id]

                slots = []
                for position in positions:
                    block_number = block_table[position // self.block_size]
                    block_offset = position % self.block_size
                    slot = block_number * self.block_size + block_offset
                    slots.append(slot)
                slot_mapping.append(slots)

                if self.sliding_window is not None:
                    sliding_window_blocks = (self.sliding_window //
                                             self.block_size)
                    block_table = block_table[-sliding_window_blocks:]
                block_tables.append(block_table)

        # vLLM uses cuda graph only for decoding requests.
        # See `capture_model` API for more details.
        batch_size = len(input_tokens)
        # record the real batch size for cudagraph in case of padding
        real_batch_size = batch_size
        max_seq_len = max(context_lens)
        use_captured_graph = (not self.model_config.enforce_eager
                              and batch_size <= _BATCH_SIZES_TO_CAPTURE[-1]
                              and max_seq_len <= self.max_seq_len_to_capture)
        if use_captured_graph:
            # Pad the input tokens, positions, and slot mapping to match the
            # batch size of the captured graph.
            graph_batch_size = _get_graph_batch_size(batch_size)
            assert graph_batch_size >= batch_size
            for _ in range(graph_batch_size - batch_size):
                input_tokens.append([])
                input_positions.append([])
                slot_mapping.append([])
                context_lens.append(0)
                block_tables.append([])
            batch_size = graph_batch_size

        max_len = max([len(t) for t in input_tokens])
        input_tokens = make_tensor_with_pad(input_tokens,
                                            max_len=max_len,
                                            pad=0,
                                            dtype=torch.long,
                                            device=self.device)
        input_positions = make_tensor_with_pad(input_positions,
                                               max_len=max_len,
                                               pad=0,
                                               dtype=torch.long,
                                               device=self.device)
        slot_mapping = make_tensor_with_pad(slot_mapping,
                                            max_len=max_len,
                                            pad=_PAD_SLOT_ID,
                                            dtype=torch.long,
                                            device=self.device)
        context_lens = torch.tensor(context_lens,
                                    dtype=torch.int,
                                    device=self.device)

        if use_captured_graph:
            # When using cuda-graph all these tensors should be
            # padded.
            assert context_lens.shape[0] == input_tokens.shape[0]
            assert context_lens.shape[0] == input_positions.shape[0]
            assert context_lens.shape[0] == slot_mapping.shape[0]

            # The shape of graph_block_tables is
            # [max batch size, max context len // block size].
            input_block_tables = self.graph_block_tables[:batch_size]
            for i, block_table in enumerate(block_tables):
                if block_table:
                    input_block_tables[i, :len(block_table)] = block_table
            block_tables = torch.tensor(input_block_tables, device=self.device)
        else:
            max_block_table_len = max(
                len(block_table) for block_table in block_tables)
            block_tables = make_tensor_with_pad(
                block_tables,
                max_len=max_block_table_len,
                pad=0,
                dtype=torch.int,
                device=self.device,
            )

        decode_metadata = self.attn_backend.make_metadata(
            is_prompt=False,
            seq_lens=None,
            seq_lens_tensor=context_lens,
            max_query_len=None,
            max_seq_len=None,
            subquery_start_loc=None,
            seq_start_loc=None,
            context_lens_tensor=None,
            block_tables=block_tables,
            use_cuda_graph=use_captured_graph,
            real_batch_size=real_batch_size,
            seq_len=input_tokens.shape[1],
            is_multi_query_mode=is_multi_query_mode,
        )
        sampling_metadata = SamplingMetadata.prepare(seq_group_metadata_list,
                                                     None, None, self.device,
                                                     self.pin_memory)
        sampling_metadata.speculate_length = self.speculate_length
        return (input_tokens, input_positions, slot_mapping, decode_metadata,
                sampling_metadata)

    def prepare_speculate_decode_input_tensors(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ):
        if self.is_driver_worker:
            (input_tokens, input_positions, slot_mapping, decode_metadata,
             sampling_metadata
             ) = self._prepare_speculate_decode(seq_group_metadata_list)
            # Broadcast the metadata.
            metadata_dict = {
                "input_tokens": input_tokens,
                "input_positions": input_positions,
                "slot_mapping": slot_mapping,
            }
            metadata_dict.update(decode_metadata.asdict_zerocopy())
            broadcast_tensor_dict(metadata_dict, src=0)
        else:
            metadata_dict = broadcast_tensor_dict(src=0)
            input_tokens = metadata_dict.pop("input_tokens")
            input_positions = metadata_dict.pop("input_positions")
            slot_mapping = metadata_dict.pop("slot_mapping")
            decode_metadata = self.attn_backend.make_metadata(**metadata_dict)
            sampling_metadata = SamplingMetadata(
                seq_groups=None,
                selected_token_indices=None,
                categorized_sample_indices=None,
                num_prompts=0,
                use_speculate=True,
                is_multi_query_mode=False,
                speculate_length=self.speculate_length,
            )
        attn_metadata = AttentionMetadata(
            num_prefills=0,
            slot_mapping=slot_mapping,
            num_prefill_tokens=0,
            num_decode_tokens=input_tokens.shape[0],
            prefill_metadata=None,
            decode_metadata=decode_metadata,
            kv_cache_dtype=self.kv_cache_dtype,
        )
        return (input_tokens, input_positions, attn_metadata,
                sampling_metadata)

    def fast_greedy_sample(self, model: torch.nn.Module,
                           hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: We always use greedy sampling for the draft model during
        # speculative decoding. In this way, we can greatly reduce the
        # sampling overhead and avoid storing draft token's logits.
        embedding_bias = None
        if hasattr(model, "lm_head"):
            lm_head_weight = model.lm_head.weight
            embedding_bias = model.lm_head.bias
        elif hasattr(model, "lm_head_weight"):
            lm_head_weight = model.lm_head_weight
        elif hasattr(model, "embed_out"):
            lm_head_weight = model.embed_out.weight
        elif hasattr(model, "embed_tokens"):
            lm_head_weight = model.embed_tokens.weight
        else:
            raise RuntimeError("Unsupported draft model")
        assert len(hidden_states.shape
                   ) == 2, "hidden_states must have shape [bs*seq_len, dim]"
        logits = torch.matmul(hidden_states, lm_head_weight.t())
        if embedding_bias is not None:
            logits += embedding_bias
        logits = tensor_model_parallel_all_gather(logits)
        logits = logits[..., :model.config.vocab_size]
        next_tokens = torch.argmax(logits, dim=-1)
        return next_tokens

    def speculate_prefill_step(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        d_kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> SpeculateOutput:
        (input_tokens, input_positions, attn_metadata, sampling_metadata,
         lora_requests, lora_mapping, multi_modal_input
         ) = self.prepare_input_tensors(seq_group_metadata_list)
        attn_metadata.slot_mapping = attn_metadata.slot_mapping.view(-1)
        input_tokens_1d = input_tokens.view(-1)
        input_positions_1d = input_positions.view(-1)
        # 1. Draft model prefill.
        # The purpose of this step is to fill draft model's kv cache
        with MarkActiveModel(ActiveModel.DRAFT):
            hidden_states = self.draft_model(
                input_ids=input_tokens_1d,
                positions=input_positions_1d,
                kv_caches=d_kv_caches,
                attn_metadata=attn_metadata,
            )
        # 2. Target model prefill
        with MarkActiveModel(ActiveModel.TARGET):
            # Execute the model.
            hidden_states = self.model(
                input_ids=input_tokens_1d,
                positions=input_positions_1d,
                kv_caches=kv_caches,
                attn_metadata=attn_metadata,
            )
            # Compute the logits.
            logits = self.model.compute_logits(hidden_states,
                                               sampling_metadata)
            # Only perform sampling in the driver worker.
            if not self.is_driver_worker:
                return None
            # Sample the next token.
            output = self.model.sample(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )
        speculate_outputs: SpeculateOutput = []
        for seq_group_output in output:
            samples = seq_group_output.samples
            assert len(
                samples
            ) == 1, "Speculative decoding only allows one seq per seq group."
            sample = samples[0]
            seq_id = sample.parent_seq_id
            token_id = sample.output_token
            logprobs_dict = sample.logprobs
            speculate_outputs.append(
                SpeculateSequenceGroupOutput(seq_id, [token_id],
                                             [logprobs_dict], 0,
                                             seq_group_output.prompt_logprobs))
        return speculate_outputs

    def speculate_decode_step(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        target_kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        d_kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> SpeculateOutput:
        input_tokens, input_positions, attn_metadata, sampling_metadata = \
            self.prepare_speculate_decode_input_tensors(seq_group_metadata_list)
        decode_metadata = attn_metadata.decode_metadata
        seq_lens_tensor = decode_metadata.seq_lens_tensor
        slot_mapping = attn_metadata.slot_mapping
        graph_batch_size = input_tokens.shape[0]
        # 1. Run the draft model
        for i in range(self.speculate_length):
            with MarkActiveModel(ActiveModel.DRAFT):
                model = self.draft_model
                model_executable = self.draft_model
                decode_metadata.is_multi_query_mode = False
                decode_metadata.seq_len = 1
                decode_metadata.seq_lens_tensor = seq_lens_tensor - self.speculate_length + i
                if decode_metadata.use_cuda_graph:
                    model_executable = self.d_graph_runners[graph_batch_size]
                    # These tensors don't need to be contiguous when
                    # using cudagraph because CUDAGraphRunner will copy
                    # the inputs to be contiguous.
                    input_tokens_1d = input_tokens[:, i]
                    input_positions_1d = input_positions[:, i]
                    attn_metadata.slot_mapping = slot_mapping[:, i]
                else:
                    input_tokens_1d = input_tokens[:, i].contiguous()
                    input_positions_1d = input_positions[:, i].contiguous()
                    attn_metadata.slot_mapping = slot_mapping[:,
                                                              i].contiguous()
                hidden_states = model_executable(
                    input_ids=input_tokens_1d,
                    positions=input_positions_1d,
                    kv_caches=d_kv_caches,
                    attn_metadata=attn_metadata,
                )
                # We always use greedy sampling to sample draft tokens.
                next_tokens = self.fast_greedy_sample(model, hidden_states)
                input_tokens[:, i + 1] = next_tokens
        # 2. Run the target model
        sampling_metadata.input_token_ids = input_tokens[:decode_metadata.
                                                         real_batch_size]
        sampling_metadata.is_multi_query_mode = True
        with MarkActiveModel(ActiveModel.TARGET):
            model = self.model
            model_executable = self.model
            decode_metadata.is_multi_query_mode = True
            decode_metadata.seq_len = self.speculate_length + 1
            input_tokens_1d = input_tokens.view(-1)
            input_positions_1d = input_positions.view(-1)
            attn_metadata.slot_mapping = slot_mapping.view(-1)
            decode_metadata.seq_lens_tensor = seq_lens_tensor
            if decode_metadata.use_cuda_graph:
                graph_runners = self.graph_runners
                # For speculative decoding, cudagraph is only used in the evaluation stage
                # for target model.
                model_executable = graph_runners[graph_batch_size]
            # Execute the model.
            hidden_states = model_executable(
                input_ids=input_tokens_1d,
                positions=input_positions_1d,
                kv_caches=target_kv_caches,
                attn_metadata=attn_metadata,
            )
            # Compute the logits.
            logits = model.compute_logits(hidden_states, sampling_metadata)
            # Only perform sampling in the driver worker.
            if not self.is_driver_worker:
                return None
            bs, num_tokens = input_tokens.shape
            # Convet logits from 1d to 2d, and slice it with real batch size.
            logits = logits.view(bs, num_tokens,
                                 -1)[:decode_metadata.real_batch_size, ...]
            # Sample the next token.
            output = self.model.sample(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )
        speculate_outputs: SpeculateOutput = []
        for seq_group_output in output:
            # 1. Handling the case when multiple tokens can be generated.
            # Here we drop the last sampled token when all draft tokens were accepted.
            # When all draft tokens were accepted, the last 2 generated tokens (the last
            # accepted draft token plus the token sampled from its logits) will miss
            # their kv caches in the draft model and requires multi-query attention.
            # Mixing single query and multi-query attention is currently not supported.
            seq_id = seq_group_output.parent_seq_id
            num_accepted = seq_group_output.num_accepted_tokens
            max_num_generated_tokens = min(num_accepted + 1,
                                           sampling_metadata.speculate_length)
            output_tokens = seq_group_output.output_tokens[:
                                                           max_num_generated_tokens]
            output_token_logprobs = seq_group_output.logprobs_list[:
                                                                   max_num_generated_tokens]
            speculate_outputs.append(
                SpeculateSequenceGroupOutput(seq_id,
                                             output_tokens,
                                             output_token_logprobs,
                                             num_accepted,
                                             prompt_logprobs=None))
        return speculate_outputs

    @torch.inference_mode()
    def speculate_execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        is_prompt: bool,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        d_kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> SpeculateOutput:
        if is_prompt:
            return self.speculate_prefill_step(seq_group_metadata_list,
                                               kv_caches, d_kv_caches)
        return self.speculate_decode_step(seq_group_metadata_list, kv_caches,
                                          d_kv_caches)

    def __del__(self) -> None:
        # Delete the CUDA graphs before deleting the pynccl communicator.
        # NOTE(woosuk): This is necessary because otherwise deadlocks can
        # happen.
        # FIXME(woosuk): This is a bit hacky. Find a more robust solution.
        # TODO(youkaichao): when we get enough user feedback that pynccl is
        # more stable than cupy, we can remove this, e.g. in v0.4.1.
        self.graph_runners.clear()
        if self.use_speculate:
            self.d_graph_runners.clear()
        self.pynccl_backend = None

    @property
    def vocab_size(self) -> int:
        return self.model_config.get_vocab_size()


class CUDAGraphRunner:

    def __init__(self, model: nn.Module):
        self.model = model
        self.input_buffers: Dict[str, torch.Tensor] = {}
        self.output_buffers: Dict[str, torch.Tensor] = {}

        self._graph: Optional[torch.cuda.CUDAGraph] = None

    @property
    def graph(self):
        assert self._graph is not None
        return self._graph

    def capture(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        memory_pool: Optional[Tuple[int, int]],
        stream: torch.cuda.Stream,
        **kwargs,
    ) -> torch.Tensor:
        assert self._graph is None
        # Run the model a few times without capturing the graph.
        # This is to make sure that the captured graph does not include the
        # kernel launches for initial benchmarking (e.g., Triton autotune).
        # Note one iteration is not enough for torch.jit.script
        for _ in range(_NUM_WARMUP_ITERS):
            self.model(
                input_ids,
                positions,
                kv_caches,
                attn_metadata,
                **kwargs,
            )
        torch.cuda.synchronize()

        # Capture the graph.
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph, pool=memory_pool, stream=stream):
            output_hidden_states = self.model(
                input_ids,
                positions,
                kv_caches,
                attn_metadata,
                **kwargs,
            )
            if hidden_states is not None:
                hidden_states.copy_(output_hidden_states)
            else:
                hidden_states = output_hidden_states
            del output_hidden_states
            # make sure `output_hidden_states` is deleted
            # in the graph's memory pool
            gc.collect()
        torch.cuda.synchronize()

        # Save the input and output buffers.
        self.input_buffers = {
            "input_ids": input_ids,
            "positions": positions,
            "kv_caches": kv_caches,
            "slot_mapping": attn_metadata.slot_mapping,
            "seq_lens_tensor": attn_metadata.decode_metadata.seq_lens_tensor,
            "block_tables": attn_metadata.decode_metadata.block_tables,
        }
        self.output_buffers = {"hidden_states": hidden_states}
        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        # KV caches are fixed tensors, so we don't need to copy them.
        del kv_caches

        # Copy the input tensors to the input buffers.
        self.input_buffers["input_ids"].copy_(input_ids, non_blocking=True)
        self.input_buffers["positions"].copy_(positions, non_blocking=True)
        self.input_buffers["slot_mapping"].copy_(attn_metadata.slot_mapping,
                                                 non_blocking=True)
        self.input_buffers["seq_lens_tensor"].copy_(
            attn_metadata.decode_metadata.seq_lens_tensor, non_blocking=True)
        self.input_buffers["block_tables"].copy_(
            attn_metadata.decode_metadata.block_tables, non_blocking=True)
        # Run the graph.
        self.graph.replay()

        # Return the output tensor.
        return self.output_buffers["hidden_states"]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def _get_graph_batch_size(batch_size: int) -> int:
    """Returns the padded batch size given actual batch size.

    Batch sizes are 1, 2, 4, _BATCH_SIZE_ALIGNMENT,
    2*_BATCH_SIZE_ALIGNMENT, 3*_BATCH_SIZE_ALIGNMENT...
    """
    if batch_size <= 2:
        return batch_size
    elif batch_size <= 4:
        return 4
    else:
        return ((batch_size + _BATCH_SIZE_ALIGNMENT - 1) //
                _BATCH_SIZE_ALIGNMENT * _BATCH_SIZE_ALIGNMENT)


def _is_block_tables_empty(block_tables: Union[None, Dict]):
    """
    Check if block_tables is None or a dictionary with all None values.
    """
    if block_tables is None:
        return True
    if isinstance(block_tables, dict) and all(
            value is None for value in block_tables.values()):
        return True
    return False
