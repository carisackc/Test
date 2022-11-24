import torch
import streamlit as st
from streamlit import components
import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, LongT5ForConditionalGeneration
import numpy as np
from math import ceil
import en_core_web_lg
from collections import Counter
from string import punctuation
# Gensim
import gensim
from gensim.summarization import summarize
import spacy

nlp = en_core_web_lg.load()

st.set_page_config(page_title ='Clinical Note Summarization', 
                   #page_icon= "Notes",
                   layout='wide')
st.title('Clinical Note Summarization')
st.sidebar.markdown('Using transformer model')

## Loading in dataset
#df = pd.read_csv('mtsamples_small.csv',index_col=0)
df = pd.read_csv("shpi_w_rouge21Nov.csv")
#df.shape
df['HADM_ID'] = df['HADM_ID'].astype(str).apply(lambda x: x.replace('.0',''))

##Renaming column
#df.rename(columns={'patient id':'Patient_ID',
#                  'hospital admission id':'Admission_ID',
#                  'transcription':'Original_Text'}, inplace = True)

#Renaming column
df.rename(columns={'SUBJECT_ID':'Patient_ID',
                  'HADM_ID':'Admission_ID',
                  'hpi_input_text':'Original_Text',
                  'hpi_reference_summary':'Reference_text'}, inplace = True)
 
 #data.rename(columns={'gdp':'log(gdp)'}, inplace=True)

#Filter selection 
st.sidebar.header("Search for Patient:")

patientid = df['Patient_ID']
patient = st.sidebar.selectbox('Select Patient ID:', patientid)
admissionid = df['Admission_ID'].loc[df['Patient_ID'] == patient]
HospitalAdmission = st.sidebar.selectbox(' ', admissionid) 

#Another way to for filter selection 
#patient = st.sidebar.multiselect(
#        "Select Patient ID:",
#        options=df['Patient_ID'].unique(),
#        default= None 
#)


#HospitalAdmission = st.sidebar.multiselect(
#        "Select Hospital Admission ID:",
#        options=df['Admission_ID'].unique(),
#        #default=df['Admission_ID'].unique()
#        default = None
#)


# List of Model available
model = st.sidebar.selectbox('Select Model', ('BART','BERT','BertGPT2','Gensim','LexRank','Long T5','Luhn','Pysummarization','SBERT Summary Tokenizer','T5','T5 Seq2Seq','T5-Base','TextRank'))


if model == 'BART':
    _num_beams = 4
    _no_repeat_ngram_size = 3
    _length_penalty = 1
    _min_length = 12
    _max_length = 128
    _early_stopping = True
else:
    _num_beams = 4
    _no_repeat_ngram_size = 3
    _length_penalty = 2
    _min_length = 30
    _max_length = 200
    _early_stopping = True
    



col3,col4 = st.columns(2) 
patientid = col3.write(f"Patient ID:  {patient} ")
admissionid =col4.write(f"Admission ID:  {HospitalAdmission} ")
    
col1, col2 = st.columns(2)
_min_length = col1.number_input("Minimum Length", value=_min_length)
_max_length = col2.number_input("Maximun Length", value=_max_length)
##_early_stopping = col3.number_input("early_stopping", value=_early_stopping)

#text = st.text_area('Input Clinical Note here')

# Query out relevant Clinical notes
original_text =  df.query(
    "Patient_ID  == @patient & Admission_ID == @HospitalAdmission"
)

original_text2 = original_text['Original_Text'].values

runtext =st.text_area('Input Clinical Note here:', str(original_text2), height=300)

reference_text = original_text['Reference_text'].values



#===== Pysummarization =====
from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.simple_tokenizer import SimpleTokenizer
from pysummarization.abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor
import regex as re

auto_abstractor = AutoAbstractor()
auto_abstractor.tokenizable_doc = SimpleTokenizer()
auto_abstractor.delimiter_list = [".", "\n"]
abstractable_doc = TopNRankAbstractor()

def pysummarizer(input_text):
#    print(type(text))
    summary = auto_abstractor.summarize(input_text, abstractable_doc)
    best_sentences=[]
    #summary_clean = ''.join([str(sentence).capitalize() for sentence in summary['summarize_result'] for summary['summarize_result'] in auto_abstractor.summarize(text, abstractable_doc)])
    for sentence in summary['summarize_result']:
        best_sentences.append(re.sub(r'\s+', ' ', sentence).strip())    
    clean_summary=''.join(sentence for sentence in best_sentences)
    return clean_summary



##===== BERT Summary tokenizer =====

def BertSummarizer(input_text):
    from transformers import BigBirdTokenizer
    from summarizer import Summarizer

    bertsummarizer = Summarizer()

    model = Summarizer()
    result = model(input_text,ratio=0.4)
    
    return result


##===== SBERT =====
from summarizer.sbert import SBertSummarizer


Sbertmodel = SBertSummarizer('paraphrase-MiniLM-L6-v2')

def Sbert(input_text):
    
#     Sbertresult = Sbertmodel(text, num_sentences=3)
    Sbertresult = Sbertmodel(input_text, ratio=0.4)
    return Sbertresult



##===== T5 Seq2Seq =====
def t5seq2seq(input_text):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
    tokenizer = AutoTokenizer.from_pretrained("t5-base")

    inputs = tokenizer("summarize: " + input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)

    summary= tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return summary

def BertGPT2(input_text):
    #import nlp

    # BioClinicalBert with BERT2GPT2 model with GPT2 decoder
    from transformers import BertTokenizer, GPT2Tokenizer, EncoderDecoderModel
    from transformers import AutoTokenizer, AutoModel

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = EncoderDecoderModel.from_pretrained("patrickvonplaten/bert2gpt2-cnn_dailymail-fp16")
    model.to(device)

    #bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    bert_tokenizer= AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    # CLS token will work as BOS token
    bert_tokenizer.bos_token = bert_tokenizer.cls_token

    # SEP token will work as EOS token
    bert_tokenizer.eos_token = bert_tokenizer.sep_token


    # make sure GPT2 appends EOS in begin and end
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        return outputs


    GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # set pad_token_id to unk_token_id -> be careful here as unk_token_id == eos_token_id == bos_token_id
    gpt2_tokenizer.pad_token = gpt2_tokenizer.unk_token


    # set decoding params
    model.config.decoder_start_token_id = gpt2_tokenizer.bos_token_id
    model.config.eos_token_id = gpt2_tokenizer.eos_token_id
    model.config.max_length = 142
    model.config.min_length = 56
    model.config.no_repeat_ngram_size = 3
    model.early_stopping = True
    model.length_penalty = 2.0
    model.num_beams = 4

    #test_dataset = nlp.load_dataset("cnn_dailymail", "3.0.0", split="test")

    batch_size = 64
 
    def Sbertmodel(batch):
        # Tokenizer will automatically set [BOS] <text> [EOS]
        # cut off at BERT max length 512
        inputs = bert_tokenizer(batch, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        input_ids = inputs.input_ids.to("cuda")
        attention_mask = inputs.attention_mask.to("cuda")

        outputs = model.generate(input_ids, attention_mask=attention_mask)

        # all special tokens including will be removed
        output_str = gpt2_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        #batch["pred"] = output_str

        return output_str
    
    Sbert(input_text)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def run_model(input_text):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model == "BART":
        bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        input_text = str(input_text)
        input_text = ' '.join(input_text.split())
        input_tokenized = bart_tokenizer.encode(input_text, return_tensors='pt').to(device)
        summary_ids = bart_model.generate(input_tokenized,
                                          num_beams=_num_beams,
                                          no_repeat_ngram_size=_no_repeat_ngram_size,
                                          length_penalty=_length_penalty,
                                          min_length=_min_length,
                                          max_length=_max_length,
                                          early_stopping=_early_stopping)

        output = [bart_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
        st.write('Summary')
        st.success(output[0])
      
    elif model == "T5":
        t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
        t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
        input_text = str(input_text).replace('\n', '')
        input_text = ' '.join(input_text.split())
        input_tokenized = t5_tokenizer.encode(input_text, return_tensors="pt").to(device)
        summary_task = torch.tensor([[21603, 10]]).to(device)
        input_tokenized = torch.cat([summary_task, input_tokenized], dim=-1).to(device)
        summary_ids = t5_model.generate(input_tokenized,
                                        num_beams=_num_beams,
                                        no_repeat_ngram_size=_no_repeat_ngram_size,
                                        length_penalty=_length_penalty,
                                        min_length=_min_length,
                                        max_length=_max_length,
                                        early_stopping=_early_stopping)
        output = [t5_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
        st.write('Summary')
        st.success(output[0])
        
      
    elif model == "Gensim": 
        output=summarize(str(input_text))
        st.write('Summary')
        st.success(output)
        
    elif model == "Pysummarization":
        output = pysummarizer(input_text)
        st.write('Summary')
        st.success(output)
        
    elif model == "BERT": 
        output = BertSummarizer(input_text)
        st.write('Summary')
        st.success(output)
        
    elif model == "SBERT Summary Tokenizer": 
        output = Sbert(input_text)
        st.write('Summary')
        st.success(output)
        
    elif model == "T5 Seq2Seq":
        output = t5seq2seq(input_text)
        st.write('Summary')
        st.success(output)
        
    elif model == "BertGPT2": #Not working correctly. to work on it later on 
        output = BertGPT2(input_text)
        st.write('Summary')
        st.success(output)


if st.button('Submit'):
    run_model(runtext)
    
   
    st.text_area('Reference text', str(reference_text))
    
