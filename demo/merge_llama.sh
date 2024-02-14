for f in "/root/autodl-tmp/lawyer-llama-13b-beta1.0/pytorch_model"*".enc"; \
    do if [ -f "$f" ]; then \
       python3 decrypt.py "$f" "/root/autodl-tmp/LLaMA-7B/consolidated.00.pth" "/root/autodl-tmp/lawyer-llama-13b-beta1.0"; \
    fi; \
done