# LLLM-LeaderboardLLM
LLLM-Leaderboard Large Language Model Paper


## Download new pwc dump 

LLLM-LeaderboardLLM/data_proccess/pwc/download_pwc_json.sh
```bash
bash download_pwc_json.sh
```

# Generate Distant labelling annotations 

LLLM-LeaderboardLLM/data_proccess/pwc_distant_label.py
```bash
# from data_process repository 
python pwc_distant_label.py --path_source pwc/dec092023/evaluation-tables.json --path_target annotations_dec092023

python pwc_distant_label.py --path_source pwc/Feb262024/evaluation-tables.json --path_target annotations_Feb262024

sbatch cpu_based_no_lb.sh python pwc_distant_label.py --path_source pwc/Feb262024/evaluation-tables.json --path_target annotations_Feb262024
```

## Download tex file from arxiv 
LLLM-LeaderboardLLMdata_proccess/download_tex_from_arxiv.sh

```bash
# from data_process repository 
bash download_tex_from_arxiv.sh

bash download_tex_from_arxiv.sh annotations_Feb262024/final_paper_links.txt arxiv_leaderboard_Feb262024

# sbatch cpu_based.sh bash download_tex_from_arxiv.sh annotations_dec092023/final_paper_links.txt arxiv_no_leaderboard_25_000
```

```bash
# from data_process repository 
ls arxiv_tex_original/ -F |grep -v / | wc -l
# 4657

ls arxiv_tex -F |grep -v / | wc -ls

ls data_proccess/arxiv_tex_dec092023/ -F |grep -v / | wc -l
ls data_proccess/arxiv_tex_summarised_dec092023/ -F |grep -v / | wc -l
ls data_proccess/arxiv_txt_summarised_dec092023/ -F |grep -v / | wc -l


ls data_proccess/arxiv_no_leaderboard_tex_25_000/ -F |grep -v / | wc -l
ls data_proccess/arxiv_no_leaderboard_tex_25_000_summarised_dec092023/ -F |grep -v / | wc -l
ls data_proccess/arxiv_no_leaderboard_txt_25_000_summarised_dec092023/ -F |grep -v / | wc -l

# 

wget --user-agent="Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.2.12) Gecko/20101026 Firefox/3.6.12" "http://arxiv.org/e-print/1704.03549v4 -P temp/ 

mv temp/"1704.03549v4" temp/"1704.03549v4.tar.gz"
tar -xf temp/"1704.03549v4.tar.gz" --directory=temp 
```

I manually removed file 1906.00121v1  as it was taking forever to convert 

```bash
2210.17517v2 

```

The script used to summarized the full latex file 

```bash
# python full_text_summarisez.py

python Dataset_creation.py --source_folder /nfs/home/kabenamualus/Research/LLLM-LeaderboardLLM/data_proccess/arxiv_tex --output_folder /nfs/home/kabenamualus/Research/LLLM-LeaderboardLLM/data_proccess/arxiv_tex_summarised_dec092023

python Dataset_creation.py --source_folder /nfs/home/kabenamualus/Research/LLLM-LeaderboardLLM/data_proccess/arxiv_no_leaderboard_tex_25_000 --output_folder /nfs/home/kabenamualus/Research/LLLM-LeaderboardLLM/data_proccess/arxiv_no_leaderboard_tex_25_000_summarised_dec092023

```


### Tex to txt

```bash
# bash tex_to_txt.sh arxiv_tex arxiv_txt
bash tex_to_txt.sh arxiv_no_leaderboard_tex_25_000_summarised_dec092023 arxiv_no_leaderboard_txt_25_000_summarised_dec092023

bash tex_to_txt.sh arxiv_tex_dec092023 arxiv_txt_dec092023

bash tex_to_txt.sh arxiv_no_leaderboard_tex_25_000 arxiv_no_leaderboard_txt_25_000_dec092023


bash tex_to_txt.sh arxiv_no_leaderboard_tex_25_000_summarised_dec092023 arxiv_no_leaderboard_txt_25_000_summarised_dec092023


```

### Text to XML

```bash

bash tex_to_xml.sh arxiv_tex_dec092023 arxiv_xml_dec092023

sbatch cpu_based.sh bash tex_to_xml.sh arxiv_tex_dec092023 arxiv_xml_lb_dec092023
sbatch cpu_based_no_lb.sh bash tex_to_xml.sh arxiv_no_leaderboard_tex_25_000 arxiv_xml_no_lb_dec092023

ls arxiv_xml_lb_dec092023 | wc -l
ls arxiv_xml_no_lb_dec092023 | wc -l
ls arxiv_xml_lb_dec092023_DOCTEAT | wc -l


```

code --diff

References:
- https://huggingface.co/blog/dpo-trl
- https://www.datacamp.com/tutorial/fine-tuning-llama-2
- https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama_2/scripts/sft_llama2.py
- https://huggingface.co/docs/peft/conceptual_guides/lora
- https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/T5/Fine_tune_CodeT5_for_generating_docstrings_from_Ruby_code.ipynb#scrollTo=RuHlP1MuR_tJ
- https://colab.research.google.com/drive/1-QZr80BN597BAtsVV3n8nrvh30HSjqN6?usp=sharing
- https://medium.com/@ajithshenoy_89727/deploy-a-fine-tune-t5-question-generation-model-using-pytorch-lightning-and-gradio-c33678bc3e88
- https://medium.com/@anchen.li/fine-tune-llama-2-with-sft-and-dpo-8b57cf3ec69
- https://github.com/mzbac/llama2-fine-tune/blob/master/generate.py
- https://www.philschmid.de/fine-tune-flan-t5 
- https://github.com/mzbac/llama2-fine-tune/blob/master/generate.py
- FLAN Template https://github.com/google-research/FLAN/blob/main/flan/templates.py 
