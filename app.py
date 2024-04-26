import streamlit as st
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

def predict_spam(text: str):
    tokenizer = BertTokenizer.from_pretrained("SpamBERT")
    model = BertForSequenceClassification.from_pretrained("SpamBERT")
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    result = classifier(text)
    return result


def main():
    st.set_page_config(page_title="Spam or Safe?")
    st.title("Spam Text Classification")
    st.write("#### A finetuned BERT model to detect spam texts.")
    st.write("")
    text = st.text_area("Enter your text here")
    if st.button("Predict"):
        result = predict_spam(text)
        if result[0]['label'] == 'LABEL_0':
            st.success(f"It's not a spam message. ({round(result[0]['score'],5) * 100}% sure)")
        elif result[0]['label'] == 'LABEL_1':
            st.error(f"IT IS A SPAM MESSAGE. ({round(result[0]['score'],5) * 100}% sure)")
    st.divider()
    st.write("""Link to the GitHub repository <a href="https://github.com/Udit-Krishna/SpamTextClassification">here</a>.""", unsafe_allow_html=True)
    
    with st.sidebar.expander("### About the model"):
        evaluation = pd.DataFrame(
            [["Loss",0.0072],
            ["Accuracy",0.9991],
            ["Precision",1.0],
            ["Recall",0.9933],
            ["F1 Score",0.9966]],
            columns=["Evaluation Criteria", "Value"]
        ).set_index("Evaluation Criteria")
        st.write("""This model is a fine-tuned version of <a href="https://huggingface.co/bert-base-uncased">BERT</a> model on <a href="https://huggingface.co/datasets/SalehAhmad/Spam-Ham">Spam-Ham</a> dataset. The model has been trained on a corpus of about 4700 rows and evaluated on around 1200 rows.""", unsafe_allow_html=True)
        st.write("It achieves the following results on the evaluation set:")
        st.table(evaluation)
        st.write(""" Find the link to the model <a href="https://huggingface.co/udit-k/HamSpamBERT">here</a>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
