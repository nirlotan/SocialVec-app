import difflib
import pickle
import urllib.request

import numpy as np
import pandas as pd
import streamlit as st
from stqdm import stqdm

from gensim.models import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile
from socialvec.socialvec import SocialVec


vector_size = 100


#############################
# Supporting Functions
# ############################

def make_clickable(link, text = ""):
    # target _blank to open new window
    # extract clickable text to display for your link
    if text == "":
        text = link  # .split('=')[1]
    return f'<a target="_blank" href="{link}">{text}</a>'



##########################
# Load Data
# #########################
@st.cache(allow_output_mutation=True)
def load_model():
    with st.spinner('Downloading SocialVec model...'):
        sv = SocialVec(model_version)

    wikidata = pd.read_csv("auxiliary/wikidata_users_sample.csv", index_col=0)
    wikidata['user_id'] = wikidata['user_id'].astype(int).astype(str)
    sv.entities['twitter_id'] = sv.entities['twitter_id'].astype(str)
    sv.entities['url'] = sv.entities['screen_name'].apply(lambda x: make_clickable("http://twitter.com/" + x, x))
    sv.entities = pd.merge(sv.entities,wikidata, left_on='twitter_id',right_on='user_id',how='left')
    sv.entities.drop('user_id',axis = 1, inplace = True)
    sv.entities['wikipedia'] = sv.entities['wikipedia'].apply(make_clickable)

    return sv
    
    

##########################
# Main
# #########################
st.set_page_config(layout="wide")
st.title("SocialVec")
st.write("Welcome to the SocialVec demo!")
st.markdown("**SocialVec** is a general framework of Social Embeddings for eliciting social world knowledge from social networks.")
st.markdown("This demo was  developed by Nir Lotan based on a research by Nir Lotan and Dr. Einat Minkov.")
st.markdown("Paper url: https://arxiv.org/abs/2111.03514; [github example](https://github.com/nirlotan/SocialVec/blob/master/examples/Basic%20Usage.ipynb); [contact](https://github.com/nirlotan)")
st.markdown("""---""")

model_version = st.sidebar.selectbox ("Model Version",[ "2020", "2020_2022"])

selected_task = st.sidebar.selectbox(
    "Select your task:",
    ["Find similar users"]
    )

data_load_state = st.text("Loading data...")
sv_model = load_model()

init_word = ""
data_load_state.text("Data Loaded Successfully!")


###########################
# side bar
# ##########################
st.markdown(
    f"""
        <style>
            .sidebar .sidebar-content {{
                width: 300px;
            }}
        </style>
        
        <style>
            .reportview-container .main .block-container{{
            max-width: {2000}px;
            padding-top: {2}rem;
            padding-right: {2}rem;
            padding-left: {2}rem;
            padding-bottom: {10}rem;
            }}
        </style>
    """,
    unsafe_allow_html=True,
)


###########################
# Main Screen
# ##########################

if selected_task == "":
    st.subheader("Please select your task on the sidebar")

elif selected_task == "Find similar users":

    c1, c2 = st.columns(2)

    user_input = c1.text_input(
        "Type a Twitter username (exact match, case sensitive):", init_word
    )
    c2.text(".")

    if c2.button("Go"):
        try:
            result_df = pd.DataFrame(
                columns=(
                    [ "Similarity", "User Name", "Name", "Description", "URL", "Wiki"]
                )
            )
            with st.empty():
                st.write("Searching... Please wait...")
                res = sv_model.get_similar(user_input,15)#ten_most_similar(user_input)
                res_html = res.to_html(escape=False)
                st.write(res_html, unsafe_allow_html=True)

        except Exception as e:
            st.write( f"Something went wrong. perhaps {user_input} is not in our database")
            st.code( f"exception: {e}")