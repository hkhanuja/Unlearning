## RMU

<img src="./images/wmdp_logo.png" width="10" /> ***Update 2024-04-23**: we simplified RMU (previously CUT) into a simpler method with similar performance. See Section 4 and Appendix B.4 of the updated paper for more details.*

1. Download the unlearn corpora (see [Datasets](#datasets))  and place into `data/`
2. Note that you will need to download the `bio-forget-corpus.jsonl` from this [Google form](https://docs.google.com/forms/d/e/1FAIpQLSdnQc8Qn0ozSDu3VE8HLoHPvhpukX1t1dIwE5K5rJw9lnOjKw/viewform)
3. Run `python3 -m rmu.unlearn` to run RMU. You can set your hyperparameters
depending on your unlearning goals. Our unlearned models with reproducible hyperparameters on WMDP-Bio and WMDP-Cyber, using `bio-forget-corpus.jsonl` and `cyber-forget-corpus.jsonl`, are available at:

- [🤗 cais/Zephyr_RMU](https://huggingface.co/cais/Zephyr_RMU) | 📙 [run_rmu_zephyr.ipynb](run_rmu_zephyr.ipynb)
- [🤗 cais/Yi-34B-Chat_RMU](https://huggingface.co/cais/Yi-34B-Chat_RMU) | 📙 [run_rmu_yi.ipynb](run_rmu_yi.ipynb)
- [🤗 cais/Mixtral-8x7B-Instruct_RMU](https://huggingface.co/cais/Mixtral-8x7B-Instruct_RMU) | 📙 [run_rmu_mixtral.ipynb](run_rmu_mixtral.ipynb)

*For evaluation, we use `lm-evaluation-harness v0.4.2`.*

## ✏️Citation
If you find this useful in your research, please consider citing our [paper]():
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
