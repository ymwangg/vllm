for d in /home/ubuntu/suzuka_models/suzuka-llama3-220M /home/ubuntu/suzuka_models/suzuka-llama3-220M-8k
    do
        for bs in 32
            do
                for spec in 1 0
                do
                    python3 benchmark.py "$d" "$bs" "$spec"
                done
            done
    done