# https://www.youtube.com/watch?v=PI8Y8-cqeIw
# https://www.section.io/engineering-education/online-machine-learning-with-river-python/
# https://github.com/discdiver/data-viz-streamlit/blob/main/app.py  <-- very good example
# https://www.youtube.com/watch?v=kXvmqg8hc70   how to deploy the app
# https://share.streamlit.io/  link to streamlit
# https://pergran1-streamlit-app-for-text-classification-app-8asu6b.streamlitapp.com/  link to the streamlit app
# this is an app for streamlit
# use the anaconda "streamlit" enviroment
import streamlit as st
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from river.naive_bayes import MultinomialNB
from river.feature_extraction import BagOfWords, TFIDF
from river.compose import Pipeline

data = pd.read_csv('fanfic_test.csv')


# convert the data into a list with sets for each observation
data = list(zip(data.story, data.rating))

# Model building
model = Pipeline(('vectorizer', BagOfWords(lowercase=True)), ('nv', MultinomialNB()))
for text, label in data:
    model = model.learn_one(text, label)

predict_dic = {'word': [], 'prediction': [], 'explicit_prob': [], 'general_prob': []}


def main():
    menu = ['Classification Model', 'About']

    choice = st.sidebar.selectbox('Selection menu', menu)
    if choice == 'Classification Model':
        st.subheader('Prediction between General and Explicit text')
        with st.form(key='mlform'):
            col1, col2 = st.columns([2, 1])
            with col1:
                message = st.text_area("Input:")
                submit_message = st.form_submit_button(label='Predict')

            with col2:
                st.write("\nWrite a text in the box and click the Predict button")
            # st.write("\nPredicts if the text ")

        if submit_message:
            prediction = model.predict_one(message)
            prediction_proba = model.predict_proba_one(message)
            probability = max(prediction_proba.values())
            # st.success("Data Submitted")

            result_col1, result_col2 = st.columns(2)
            with result_col1:
                st.info('Original Text')
                st.write(message)

                # Write about each word removed
                message_list = message.split()  # split the sentence

                for index, word in enumerate(message_list):
                    remove_list = message_list.copy()  # copy the list of the original sentence
                    removed_word = remove_list.pop(index)
                    new_word = " ".join(remove_list)
                    new_pred = model.predict_one(new_word)
                    # st.write(f"The removed word is: {removed_word} and the whole sentence is : {new_word} and the prediction is: {new_pred}")
                    prob_dic = model.predict_proba_one(new_word)
                    predict_dic['word'].append(removed_word)
                    predict_dic['prediction'].append(new_pred)
                    predict_dic['explicit_prob'].append(prob_dic['explicit'])
                    predict_dic['general_prob'].append(prob_dic['general'])

                # create a dataframe from the dic
                df = pd.DataFrame.from_dict(predict_dic)
                # st.table(df)
                # st.dataframe(df)

                with result_col2:
                    st.info('Prediction')
                    st.write(prediction)

                # st.info('Probability')
                # s  st.write(prediction_proba)

                # plot of the probability

                # st.dataframe(df_proba)
        col_test = st.container()
        with col_test:
            try:

                df_proba = pd.DataFrame({'label': prediction_proba.keys(), 'probability': prediction_proba.values()})
                st.subheader('Chart over probability')
                fig = alt.Chart(df_proba).mark_bar().encode(x='label', y='probability')
                st.altair_chart(fig, use_container_width=True)

                # st.info("This is a new column")
                # st.success('New test')

                st.subheader('Explore how each word affect the model')
                #st.write(prediction)
                #st.table(df)

                df['explicit_contribution'] = prediction_proba['explicit'] - df.explicit_prob

                #st.write(df.head())
                sortLevel = df.word
                #fig = alt.Chart(df).mark_bar().encode(y='word', x=alt.X('explicit_contribution'),
                                                  #    color='prediction')
               # st.altair_chart(fig, use_container_width=True)


                fig2 = plt.figure(figsize=(10, 4))

                sns.barplot(x="explicit_contribution", y  = 'word', hue = 'prediction', data=df)
                st.write(
                    """
                    The plot below shows how each words affect the probability of the text being Explicit. 
                    
                    The colors shows what the text would be classified as if that one word was omitted from the text. 
                    
                    #### Example:
                    The text **"You look like a sexy man"** is classified as Explicit, and when the word **"sexy"**
                    is removed so only **"You look lika a man"** is left than the text is classified as Generic, meaning 
                    that **"sexy"** had a large impact on the classification. 
                    """
                )
                st.pyplot(fig2)

                # fig, ax = plt.subplots()
                # plt.bar(data=df, x='word', y='explicit_contribution')
                # st.pyplot(fig)

            except:
                pass
            # st.dataframe(df)
            # Viz






    #elif choice == 'Manage':
        #st.subheader('Manage & Monitor Results')
        #st.info("You just came here so big congrats!")

    else:
        st.write(
            """
            ## Hello! 🙌
            
            This is a fun little project I created after writing code to extract fanfiction data from a website. 
            The original idea with the data was to create a model that would write fanfiction on its own by learning from the data.
            But I noticed that a classification model that would classify between a "general" text or text that is "explicit" (have dirty words) 🙈 would
            also be a fun challenge.  

            I also wanted to explore the black box problem when creating deep learning models, I therefore created an algorithm that would classify the same text but
            one word would be removed one step at a time. By doing this I would be able to see which word that had the biggest impact on the classification. 
            
            A model like this would be able to filter out text or warn young users that the message or text they are about to see might contain language that are inappropriate. 
            
            The GitHub repository with the code can be found [here](https://github.com/pergran1/Streamlit-app-for-text-classification)
            
            My LinkedIn is found [here](https://www.linkedin.com/in/per-granberg-579969108/)


            """
        )


if __name__ == '__main__':
    main()
