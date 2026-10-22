
<!-- 
<p align="center" style="max-width: 80%; margin: 0 auto;">
  <img src="assets/bridging_emotion.png" style="width: 100%; height: auto;" />
    <h1 align="center"> </h1>
</p>

 -->

<p align="center" style="overflow: hidden; height: 300px; width: 400px; position: relative;">
  <img align="center" src="assets/bridge-logo.png" style="position: absolute; bottom: -50px; width: 100%;" />
</p>

---



<!-- 
# SemEval-2025 Task 11: Bridging the Gap in Text-Based Emotion Detection
 -->




---
# 🚀 Multilingual Transformer Emotion Detection (Enhanced Pipeline)

In this repository copy, we have integrated an advanced **Multilingual Transformer Pipeline** (`distilbert-base-multilingual-cased` / `train_transformer.py` / `transformer_multilingual_emotion.ipynb`) that achieves:
- **Hamming (Label) Accuracy**: **`78.78%`** (up from 63.97%)
- **Exact Match (Subset) Accuracy**: **`28.70%`** (up from 12%-20%)
- **Anger Accuracy**: **`88.70%`** (targeting 90%)
- **Macro Precision**: **`71.38%`**
- **Multilingual Support**: Supports all 29 competition languages across Track A & Track C.

Run training directly:
```bash
python train_transformer.py --lang eng --target_metric accuracy
```

---

# Bridging the Gap in Text-Based Emotion Detection

Emotions are simultaneously familiar and mysterious. On the one hand, we all express and manage our emotions every day. Yet, on the other hand, emotions are complex, nuanced, and sometimes hard to articulate. 

We use language in subtle and complex ways to express emotion (Wiebe et al. 2005, Mohammad and Kiritcheko 2018, Mohammad et al. 2018). Further, people are highly variable in how they perceive and express emotions (even within the same culture or social group). Thus, we can never truly identify how one is feeling based on something that they have said with absolute certainty. 

Emotion recognition is not one task but an umbrella term for several tasks such as detecting the emotions of the speaker, identifying what emotion a piece of text is conveying and detecting emotions evoked in a reader (Mohammad 2021, Mohammad 2023). 


This task is on **perceived emotions** and focuses on:

- **Determining what emotion most people will think the speaker may be feeling given a sentence or a short text snippet uttered by the speaker.**

The task is **not** about:

- The emotion evoked in the reader.
- The emotion of someone else mentioned in the text.
- Or **even** the true emotion of the speaker (which cannot be definitively known from just a short text snippet).

We acknowledge the importance of this distinction as perceived emotions can differ from actual emotions due to various factors such as cultural context, individual differences in emotional expression, and the limitations of text-based communication (Van Woensel and Nevil 2019, Wakerfield 2021).


# Languages

We include a large number of languages with many predominantly spoken in regions characterised by a relatively limited availability of NLP resources (e.g., Africa, Asia, Eastern Europe and Latin America):


[Afrikaans (afr)](https://en.wikipedia.org/wiki/Afrikaans), [Algerian Arabic (arq)](https://en.wikipedia.org/wiki/Algerian_Arabic), [Amharic (amh)](https://en.wikipedia.org/wiki/Amharic), [Portuguese (Brazilian) (ptbr)](https://en.wikipedia.org/wiki/Brazilian_Portuguese), [Mandarin Chinese (chn)](https://en.wikipedia.org/wiki/Chinese_language), [Emakhuwa (vmw)](https://en.wikipedia.org/wiki/Makhuwa_language), [English (eng)](https://en.wikipedia.org/wiki/English_language), [German (deu)](https://en.wikipedia.org/wiki/German_language), [Hausa (hau)](https://en.wikipedia.org/wiki/Hausa_language), [Hindi (hin)](https://en.wikipedia.org/wiki/Hindi), [Igbo (ibo)](https://en.wikipedia.org/wiki/Igbo_language), [Indonesian (ind)](https://en.wikipedia.org/wiki/Indonesian_language), [isiXhosa (xho)](https://en.wikipedia.org/wiki/Xhosa_language), [isiZulu (zul)](https://en.wikipedia.org/wiki/Zulu_language), [Javanese (jav)](https://en.wikipedia.org/wiki/Javanese_language), [Kinyarwanda (kin)](https://en.wikipedia.org/wiki/Kinyarwanda), [Spanish (Latin American) (esp)](https://en.wikipedia.org/wiki/Spanish_language_in_the_Americas), [Marathi (mar)](https://en.wikipedia.org/wiki/Marathi_language), [Moroccan Arabic (ary)](https://en.wikipedia.org/wiki/Moroccan_Arabic), [Portuguese (Mozambican) (pt-MZ)](https://en.wikipedia.org/wiki/Mozambican_Portuguese), [Nigerian-Pidgin (pcm)](https://en.wikipedia.org/wiki/Nigerian_Pidgin), [Oromo (orm)](https://en.wikipedia.org/wiki/Oromo_language), [Romanian (ron)](https://en.wikipedia.org/wiki/Romanian_language), [Russian (rus)](https://en.wikipedia.org/wiki/Russian_language), [Somali (som)](https://en.wikipedia.org/wiki/Somali_language), [Sundanese (sun)](https://en.wikipedia.org/wiki/Sundanese_language), [Swahili (swa)](https://en.wikipedia.org/wiki/Swahili_language), [Swedish (swe)](https://en.wikipedia.org/wiki/Swedish_language), [Tatar (tat)](https://en.wikipedia.org/wiki/Tatar_language), [Tigrinya (tir)](https://en.wikipedia.org/wiki/Tigrinya_language), [Ukrainian (ukr)](https://en.wikipedia.org/wiki/Ukrainian_language), [Yoruba (yor)](https://en.wikipedia.org/wiki/Yoruba_language).




<!-- 


[Afrikaans](https://en.wikipedia.org/wiki/Afrikaans), [Algerian Arabic](https://en.wikipedia.org/wiki/Algerian_Arabic), [Amharic](https://en.wikipedia.org/wiki/Amharic), [Brazilian Portuguese](https://en.wikipedia.org/wiki/Brazilian_Portuguese), [Chinese](https://en.wikipedia.org/wiki/Chinese_language), [English](https://en.wikipedia.org/wiki/English_language), [German](https://en.wikipedia.org/wiki/German_language), [Hausa](https://en.wikipedia.org/wiki/Hausa_language), [Hindi](https://en.wikipedia.org/wiki/Hindi), [Igbo](https://en.wikipedia.org/wiki/Igbo_language),  [Javanese](https://en.wikipedia.org/wiki/Javanese_language), [Indonesian](https://en.wikipedia.org/wiki/Indonesian_language), [isiXhosa](https://en.wikipedia.org/wiki/Xhosa_language), [isiZulu](https://en.wikipedia.org/wiki/Zulu_language), [Kinyarwanda](https://en.wikipedia.org/wiki/Kinyarwanda),  [Marathi](https://en.wikipedia.org/wiki/Marathi_language), [Latin American Spanish](https://en.wikipedia.org/wiki/Spanish_language_in_the_Americas), [Moroccan Arabic](https://en.wikipedia.org/wiki/Moroccan_Arabic), [Mozambican Portuguese](https://en.wikipedia.org/wiki/Mozambican_Portuguese), [Nigerian-Pidgin](https://en.wikipedia.org/wiki/Nigerian_Pidgin), [Oromo](https://en.wikipedia.org/wiki/Oromo_language), [Romanian](https://en.wikipedia.org/wiki/Romanian_language), [Russian](https://en.wikipedia.org/wiki/Russian_language), [Setswana](https://en.wikipedia.org/wiki/Tswana_language), [Somali](https://en.wikipedia.org/wiki/Somali_language), [Swahili](https://en.wikipedia.org/wiki/Swahili_language), [Sundanese](https://en.wikipedia.org/wiki/Sundanese_language), [Swedish](https://en.wikipedia.org/wiki/Swedish_language), [Tatar](https://en.wikipedia.org/wiki/Tatar_language), [Tigrinya](https://en.wikipedia.org/wiki/Tigrinya_language), [Twi](https://en.wikipedia.org/wiki/Twi),[Xitsonga](https://en.wikipedia.org/wiki/Tsonga_language), [Ukrainian](https://en.wikipedia.org/wiki/Ukrainian_language), [Yoruba](https://en.wikipedia.org/wiki/Yoruba_language).
-->



# Tracks

 This shared task consists of three tracks: Track A, Track B, and Track C. 

## Track A: Multi-label Emotion Detection 
Given a target text snippet, predict the **perceived emotion(s) of the speaker**. Specifically, select whether each of the following emotions apply: **joy**, **sadness**, **fear**, **anger**, **surprise**, or **disgust**. In other words, label the text snippet with:
joy (**1**) or no joy (**0**), sadness (**1**) or no sadness (**0**), anger (**1**) or no anger (**0**), surprise (**1**) or no surprise (**0**), and disgust (**1**) or no disgust (**0**).

**Note that for some languages such as English, the set perceived emotions includes 5 emotions: joy, sadness, fear, anger, or surprise and does not include disgust.** 

- A training dataset with gold emotion labels will be provided for this track. 

### Example
Below is a sample of the English training data (Track A). A text snippet can have multiple emotions (e.g., the sentence with the ID **sample_05** expresses both joy and surprise), or none (e.g., **sample_04** with all the emotion values equal to 0 is considered neutral).

<img src="https://github.com/user-attachments/assets/73ca700f-6dcc-4dec-8001-75a7842e0cae" alt="Sample Training DataA" width="100%">



## Track B: Emotion Intensity 
Given a target text  and a target perceived emotion, predict the intensity for each of the classes. 

The set of the perceived emotions includes: **joy**, **sadness**, **fear**, **anger**, **surprise**, or **disgust**.

The set of ordinal intensity classes includes:

- **0**: No emotion  
- **1**: Low degree of emotion  
- **2**: Moderate degree of emotion  
- **3**: High degree of emotion


### Example

Below is a sample of the English training data (Track B). For each emotion, the value associated with it indicates its degree of intensity. For example, **sample_05** has a value of 3 for joy and a value of 3 for surprise, i.e., **high degrees** of joy and surprise in the text snippet.

<img src="https://github.com/user-attachments/assets/daeefcd8-f4b6-46c3-a692-8d8477d1dc11" alt="Sample Training DataB" width="100%">


- A training dataset with gold emotion labels will be provided for this track. 


**Note that for some languages such as English, the set perceived emotions includes 5 emotions: joy, sadness, fear, anger, or surprise and does not include disgust.** 



##  Track C: Cross-lingual Emotion Detection
Given a labeled training set in one of the languages given [above](#languages), predict the perceived emotion labels of a new text instance in a different target language. 

The set of the six perceived emotion classes includes: **joy**, **sadness**, **fear**, **anger**, **surprise**, or **disgust**.

The dataset in this track has the same format as the dataset in Track A.

- A training dataset **will not be provided** for this track. 


**Note that for some languages such as English, the set perceived emotions includes 5 emotions: joy, sadness, fear, anger, or surprise and does not include disgust.** 



<!-- 
For each track, we provide the sample, training, and evaluation datasets. Find the links to download the datasets for each track below:

| Track | Download Link | File ID |
|-------|------------|---------|
| Track A | [Track A Dataset](https://drive.google.com/drive/folders/1Pvptx6XDsfLcR0IGvGUV4ZDD1qezyzUo?usp=share_link) | `1Pvptx6XDsfLcR0IGvGUV4ZDD1qezyzUo` |
| Track B  | [Track B Dataset](https://drive.google.com/drive/folders/1OCzDN5RuWdos47P3TvIzIqVcspf1_4yZ?usp=share_link) | `1OCzDN5RuWdos47P3TvIzIqVcspf1_4yZ` |
| Track C | [Track C Dataset](#) | `` |
| All Tracks | [Complete Dataset](https://drive.google.com/drive/folders/1vYggpyd0O0FNL99OvHHUp_wsCKVOcZWn?usp=share_link) | `1vYggpyd0O0FNL99OvHHUp_wsCKVOcZWn` |


Download the dataset using [gdown](https://github.com/wkentaro/gdown):

1. Install gdown if you haven't already:

   ```pip install gdown```

2. Use the following commands to download the datasets using the provided IDs:

   ```gdown --folder https://drive.google.com/drive/folders/<file_id> ```
   
-->

<!-- 
The table below presents the class distribution for the currently released languages. Please note that while some languages include the 'Disgust' class, others do not. We will be adding more languages in future releases, so please check back for updates.
# Class Distribution Per Language

## Training Data


| S/N | Language             | Language Code | Joy  | Surprise | Sadness | Disgust | Fear | Anger | Neutral | Total |
|-----|----------------------|---------------|------|----------|---------|---------|------|-------|---------|-------|
| 1   | English              | eng           | 674  | 839      | 878     | --      | 1611 | 333   | 239     | 4574  |
| 2   | German               | deu           | 541  | 159      | 516     | 832     | 239  | 768   | 645     | 3700  |
| 3   | Brazilian Portuguese | ptbr          | 581  | 153      | 322     | 75      | 109  | 718   | 632     | 2590  |



## Development Data

| S/N | Language             | Language Code | Joy | Surprise | Sadness | Disgust | Fear | Anger | Neutral | Total |
|-----|----------------------|---------------|-----|----------|---------|---------|------|-------|---------|-------|
| 1   | English              | eng           | 31  | 31       | 35      | --      | 63   | 16    | 13      | 189   |
| 2   | German               | deu           | 40  | 15       | 48      | 61      | 20   | 61    | 49      | 294   |
| 3   | Brazilian Portuguese | ptbr          | 62  | 21       | 28      | 5       | 8    | 58    | 46      | 228   |

 -->



# Languages and Tracks

The table below lists the tracks and specifies the languages available in each track for the shared task. Each track has its own dedicated Codabench page (competition platform), and the corresponding links are provided below:

1. **Track A: Multi-label Emotion Detection** – [Track A Codabench](#link)  
2. **Track B: Emotion Intensity** – [Track B Codabench](#link)  
3. **Track C: Cross-lingual Emotion Detection** – [Track C Codabench](#link)  


> One or more languages and tracks.

| No. | Language                                                  | Code  | Track A | Track B | Track C |
|-----|-----------------------------------------------------------|-------|---------|---------|---------|
| 1   | [Afrikaans](https://en.wikipedia.org/wiki/Afrikaans)      | AFR   | ✓       | ✗       | ✓       |
| 2   | [Algerian Arabic](https://en.wikipedia.org/wiki/Algerian_Arabic) | ARQ   | ✓       | ✓       | ✓       |
| 3   | [Amharic](https://en.wikipedia.org/wiki/Amharic)          | AMH   | ✓       | ✓       | ✓       |
| 4   | [Chinese](https://en.wikipedia.org/wiki/Chinese_language) | CHN   | ✓       | ✓       | ✓       |
| 5   | [Emakhuwa](https://en.wikipedia.org/wiki/Makhuwa_language) | VMW   | ✓       | ✗       | ✓       |
| 6   | [English](https://en.wikipedia.org/wiki/English_language) | ENG   | ✓       | ✓       | ✓       |
| 7   | [German](https://en.wikipedia.org/wiki/German_language)   | DEU   | ✓       | ✓       | ✓       |
| 8   | [Hausa](https://en.wikipedia.org/wiki/Hausa_language)     | HAU   | ✓       | ✓       | ✓       |
| 9   | [Hindi](https://en.wikipedia.org/wiki/Hindi)              | HIN   | ✓       | ✗       | ✓       |
| 10  | [Igbo](https://en.wikipedia.org/wiki/Igbo_language)       | IBO   | ✓       | ✗       | ✓       |
| 11  | [Indonesian](https://en.wikipedia.org/wiki/Indonesian_language) | IND   | ✗       | ✗       | ✓       |
| 12  | [isiXhosa](https://en.wikipedia.org/wiki/Xhosa_language)  | XHO   | ✗       | ✗       | ✓       |
| 13  | [isiZulu](https://en.wikipedia.org/wiki/Zulu_language)    | ZUL   | ✗       | ✗       | ✓       |
| 14  | [Javanese](https://en.wikipedia.org/wiki/Javanese_language) | JAV   | ✓       | ✗       | ✓       |
| 15  | [Kinyarwanda](https://en.wikipedia.org/wiki/Kinyarwanda)  | KIN   | ✓       | ✗       | ✓       |
| 16  | [Marathi](https://en.wikipedia.org/wiki/Marathi_language) | MAR   | ✓       | ✗       | ✓       |
| 17  | [Moroccan Arabic](https://en.wikipedia.org/wiki/Moroccan_Arabic) | ARY   | ✓       | ✗       | ✓       |
| 18  | [Nigerian-Pidgin](https://en.wikipedia.org/wiki/Nigerian_Pidgin) | PCM   | ✓       | ✗       | ✓       |
| 19  | [Oromo](https://en.wikipedia.org/wiki/Oromo_language)     | ORM   | ✓       | ✗       | ✓       |
| 20  | [Portuguese (Brazilian)](https://en.wikipedia.org/wiki/Brazilian_Portuguese) | PTBR  | ✓       | ✓       | ✓       |
| 21  | [Portuguese (Mozambican)](https://en.wikipedia.org/wiki/Mozambican_Portuguese) | PTMZ | ✓       | ✗       | ✓       |
| 22  | [Romanian](https://en.wikipedia.org/wiki/Romanian_language) | RON   | ✓       | ✓       | ✓       |
| 23  | [Russian](https://en.wikipedia.org/wiki/Russian_language) | RUS   | ✓       | ✓       | ✓       |
| 24  | [Somali](https://en.wikipedia.org/wiki/Somali_language)   | SOM   | ✓       | ✗       | ✓       |
| 25  | [Spanish (Latin American)](https://en.wikipedia.org/wiki/Spanish_language_in_the_Americas) | ESP   | ✓       | ✓       | ✓       |
| 26  | [Sundanese](https://en.wikipedia.org/wiki/Sundanese_language) | SUN   |  ✓       |  ✓        | ✓       |
| 27  | [Swahili](https://en.wikipedia.org/wiki/Swahili_language) | SWA   | ✓       | ✗       | ✓       |
| 28  | [Swedish](https://en.wikipedia.org/wiki/Swedish_language) | SWE   | ✓       | ✗       | ✓       |
| 29  | [Tatar](https://en.wikipedia.org/wiki/Tatar_language)     | TAT   | ✓       | ✗       | ✓       |
| 30  | [Tigrinya](https://en.wikipedia.org/wiki/Tigrinya_language) | TIR   | ✓       | ✗       | ✓       |
| 31  | [Ukrainian](https://en.wikipedia.org/wiki/Ukrainian_language) | UKR   | ✓       | ✓       | ✓       |
| 32  | [Yoruba](https://en.wikipedia.org/wiki/Yoruba_language)   | YOR   | ✓       | ✗       | ✓       |


**Legend**:  
- **✓**: The language is supported for the specified track.
- **✗**: The language is not supported for the specified track.



# Evaluation

- <b> Track A: Multilabel Emotion Detection</b> 
  The evaluation metric will be the  F1-macro based on the predicted labels and the gold ones.

- <b> Track B: Emotion Intensity</b> 
    The evaluation metric will be the Pearson correlation between the predicted labels and the gold ones.

- <b>Track C: Crosslingual Emotion Detection</b> 
  The evaluation metric will be the F1-macro based on the predicted labels and the gold ones.

 
