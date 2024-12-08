## RMU

RMU (representation misdirection for unlearnin)  is inspired by the representation engineering, based on the idea that model representations encode knowledge of the world and that these representations can be manipulated to affect model behavior. Specifically, this approach proposes a two-part loss function with a forget loss and a retain loss to change model activations on hazardous data to a random direction while preserving model activations on benign data. <br/>

For unlearning alpaca-7b, 

1. Use the unlearn corpora under `data/SafeRLHF-corpora` (source: [🤗 PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF))
2. Use the alpaca-7b model (source: [🤗 PKU-Alignment/alpaca-7b-reproduced](https://huggingface.co/PKU-Alignment/alpaca-7b-reproduced))
3. Run `python3 -m rmu.unlearn` to run RMU. You can set your hyperparameters depending on your unlearning goals. Our unlearned models with reproducible hyperparameters (penalizing weight on retain loss $\alpha = 0,1,10,100,1000$) on SafeRLHF data , using `Privacy_Violation_train.jsonl` as the forget set and `safe_train_sampled.jsonl` as the retain set, are available at:

- [🤗 lumiereljy/alpaca-rmu-alpha-0](https://huggingface.co/lumiereljy/alpaca-rmu-alpha-0) | 📙 [run_rmu_alpaca.ipynb](run_rmu_alpaca.ipynb)
- [🤗 lumiereljy/alpaca-rmu-alpha-1](https://huggingface.co/lumiereljy/alpaca-rmu-alpha-1) | 📙 [run_rmu_alpaca.ipynb](run_rmu_alpaca.ipynb)
- [🤗 lumiereljy/alpaca-rmu-alpha-10](https://huggingface.co/lumiereljy/alpaca-rmu-alpha-10) | 📙 [run_rmu_alpaca.ipynb](run_rmu_alpaca.ipynb)
- [🤗 lumiereljy/alpaca-rmu-alpha-100](https://huggingface.co/lumiereljy/alpaca-rmu-alpha-100) | 📙 [run_rmu_alpaca.ipynb](run_rmu_alpaca.ipynb)
- [🤗 lumiereljy/alpaca-rmu-alpha-1000](https://huggingface.co/lumiereljy/alpaca-rmu-alpha-1000) | 📙 [run_rmu_alpaca.ipynb](run_rmu_alpaca.ipynb)
    




## ✏️Citation
If you find this useful in your research, please consider citing [paper]():
```
@misc{li2024wmdp,
      title={The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning}, 
      author={Nathaniel Li and Alexander Pan and Anjali Gopal and Summer Yue and Daniel Berrios and Alice Gatti and Justin D. Li and Ann-Kathrin Dombrowski and Shashwat Goel and Long Phan and Gabriel Mukobi and Nathan Helm-Burger and Rassin Lababidi and Lennart Justen and Andrew B. Liu and Michael Chen and Isabelle Barrass and Oliver Zhang and Xiaoyuan Zhu and Rishub Tamirisa and Bhrugu Bharathi and Adam Khoja and Zhenqi Zhao and Ariel Herbert-Voss and Cort B. Breuer and Samuel Marks and Oam Patel and Andy Zou and Mantas Mazeika and Zifan Wang and Palash Oswal and Weiran Liu and Adam A. Hunt and Justin Tienken-Harder and Kevin Y. Shih and Kemper Talley and John Guan and Russell Kaplan and Ian Steneker and David Campbell and Brad Jokubaitis and Alex Levinson and Jean Wang and William Qian and Kallol Krishna Karmakar and Steven Basart and Stephen Fitz and Mindy Levine and Ponnurangam Kumaraguru and Uday Tupakula and Vijay Varadharajan and Yan Shoshitaishvili and Jimmy Ba and Kevin M. Esvelt and Alexandr Wang and Dan Hendrycks},
      year={2024},
      eprint={2403.03218},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
