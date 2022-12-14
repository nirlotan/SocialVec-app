import difflib
import pickle
import urllib.request

import numpy as np
import pandas as pd
import streamlit as st
from gensim.models import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile

vector_size = 100


#############################
# Supporting Functions
# ############################


def id_to_name(uid):

    res_name = ud_df[ud_df["user_id"] == uid]["screen_name"].to_string(index=False).strip()
    if res_name == "Series([], )":
        with open("missing.log", "a") as myfile:
            myfile.write(uid + "\n")
        return str(uid)
    else:
        return res_name


def name_to_id(name):
    uid = ud_df[ud_df["screen_name"] == name]["user_id"].to_string(index=False)
    return uid.strip()


def tsnescatterplot(model, word, list_names):
    arrays = np.empty((0, vector_size), dtype="f")
    word_labels = [word]

    # adds the vector of the query word
    arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)

    # gets list of most similar words
    close_words = model.wv.most_similar([word], topn=50)

    # adds the vector for each of the closest words to the array
    for wrd_score in close_words:
        wrd_vector = model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        arrays = np.append(arrays, wrd_vector, axis=0)

    #    word_labels = [ud_df[ud_df['user_id']==wrd]['screen_name'].to_string(index=False).strip() for wrd in word_labels]

    word_output = []
    for wrd in word_labels:
        user_name = (
            ud_df[ud_df["user_id"] == wrd]["screen_name"].to_string(index=False).strip()
        )
        if user_name != "Series([], )":
            word_output.append(user_name)
        else:
            with open("missing.log", "a") as myfile:
                myfile.write(wrd + "\n")

    return word_output


def ten_most_similar(wrd):
    uid = name_to_id(wrd)
    # uid = ud_df[ud_df['screen_name']==wrd]['user_id'].to_string(index=False)
    # uid = uid.strip()
    return tsnescatterplot(
        sv_model,
        uid,
        [i[0] for i in sv_model.wv.most_similar(negative=[uid], topn=30)],
    )


def analogy(namea, nameb, namec):

    ida = name_to_id(namea)
    idb = name_to_id(nameb)
    idc = name_to_id(namec)

    if ida == "Series([], )":
        st.write(f"User named {namea} is not in my database")
    if idb == "Series([], )":
        st.write(f"User named {nameb} is not in my database")
    if idc == "Series([], )":
        st.write(f"User named {namec} is not in my database")

    result = sv_model.most_similar(negative=[ida], positive=[idb, idc], topn=5)

    return result


def make_clickable(link):
    # target _blank to open new window
    # extract clickable text to display for your link
    text = link  # .split('=')[1]
    return f'<a target="_blank" href="{link}">{text}</a>'



##########################
# Load Data
# #########################
@st.cache(allow_output_mutation=True)
def load_data():
    ud_df = pd.read_pickle("auxiliary/users_with_over_200_DETAILS.pkl")
    wikipedia = pd.read_csv("auxiliary/wikidata_users_sample.csv")
    wikipedia.user_id = wikipedia.user_id.apply(lambda x: int(x))
    wikipedia.user_id = wikipedia.user_id.astype(str)
    ud_df = pd.merge(ud_df, wikipedia, on="user_id", how="outer")

    return ud_df


@st.cache(allow_output_mutation=True)
def load_model():
    with st.spinner('Downloading word2vec model... please hold...'):
        sg_2020_2022 = pickle.load(urllib.request.urlopen("https://www.dropbox.com/s/qiuqdigicuxsavz/SocialVec2020_2022.pkl?dl=1"))
        
  
    return sg_2020_2022
    
    

##########################
# Main
# #########################
st.set_page_config(layout="wide")
st.title("SocialVec")
st.write("Welcome to the SocialVec demo!")
st.markdown("**SocialVec** is a general framework of Social Embeddings for eliciting social world knowledge from social networks.")
st.markdown("This demo was  developed by Nir Lotan based on a research by Nir Lotan and Dr. Einat Minkov.")
st.markdown("Paper url: https://arxiv.org/abs/2111.03514; [github example](https://github.com/nirlotan/SocialVec); [contact](https://github.com/nirlotan)")
st.markdown("""---""")

selected_task = st.sidebar.selectbox(
    "Select your task:",
    (   "Find similar users", 
        "Analogy game", 
        "Who is closer to who?",
        "Find Similar for 3 users")
    )
show_search = st.sidebar.checkbox("Show Search Engine")
data_load_state = st.text("Loading data...")
ud_df = load_data()
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


if show_search == True:

    st.markdown(
        f"""
            <style>
                .sidebar .sidebar-content {{
                    width: 475px;
                }}
            </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.subheader("Search twitter users")
    user_input = st.sidebar.text_input("Search for:")
    string_list = ud_df.sort_values(by="followers_count", ascending=False)[
        "screen_name"
    ].to_list()[0:22000]
    search_res = difflib.get_close_matches(user_input, string_list, 5)

    df_display = pd.DataFrame()
    for name in search_res:
        df_display = df_display.append(
            (ud_df[ud_df["screen_name"] == name][["screen_name", "description"]])
        )

    if search_res:
        df_display = df_display.assign(hack="").set_index("hack")
        st.sidebar.table(df_display[["screen_name", "description"]])

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
                res = ten_most_similar(user_input)
                for username in res:
                    if username != "Series([], )":
                        desc = ud_df[ud_df["screen_name"] == username][
                            "description"
                        ].to_string(index=False)
                        name = ud_df[ud_df["screen_name"] == username][
                            "name"
                        ].to_string(index=False)
                        wiki = ud_df[ud_df["screen_name"] == username][
                            "wikipedia"
                        ].to_string(index=False)
                        url = "http://twitter.com/" + username

                        original_user_id = name_to_id(user_input)
                        checked_user_id = name_to_id(username)

                        simil = sv_model.wv.similarity(original_user_id, checked_user_id)

                        result_df = result_df.append(
                            {
                                "Similarity": float("{:.3f}".format(simil)),
                                "User Name": username,
                                "Name": name,
                                "Description": desc,
                                "URL": url,
                                "Wiki": wiki
                            },
                            ignore_index=True,
                        )

                    else:
                        continue
                st.write("10-20 closest users to " + user_input + " are:")

        except Exception as e:
            st.write( f"Something went wrong. perhaps {user_input} is not in our database")
            st.code( f"exception: {e}")
    
        # link is the column with hyperlinks
        result_df["URL"] = result_df["URL"].apply(make_clickable)
        result_df["Wiki"] = result_df["Wiki"].apply(make_clickable)
        result_df_html = result_df.to_html(escape=False)
        st.write(result_df_html, unsafe_allow_html=True)


if selected_task == "Analogy game":

    st.write("Write three Twitter user names (exact names).")
    st.write("we take the analogy of the first two users and apply on the third")

    c1, c2, c3, c4 = st.columns(4)

    user1 = c1.text_input("Twitter user: ", "")
    user2 = c2.text_input("is to:", "")
    user3 = c3.text_input("like:", "")
    try:
        res = ""
        if st.button("Go"):
            analogy_result = analogy(user1, user2, user3)
            top_res = id_to_name(analogy_result[0][0])
            st.write(user1 + " is to " + user2 + " like " + user3 + " is to " + top_res)
        c4.text_input("is to:", top_res)
        st.subheader('Top 5 results:')
        for i, item in enumerate(analogy_result):
            st.write(f"{i+1}. {id_to_name(item[0])}")
    except:
        st.write("")

        
if selected_task == "Who is closer to who?":

    st.write("Write three Twitter user names (exact names).")
    st.write("we take the analogy of the first two users and apply on the third")

    user1 = st.text_input("Main user: ", "")
    c1, c2 = st.columns(2)
    user2 = c1.text_input("Compare to 1:", "")
    user3 = c2.text_input("Compare to 2:", "")
    try:
        res = ""
        if st.button("Go"):
            simil1 = sv_model.similarity(name_to_id(user1), name_to_id(user2))
            simil2 = sv_model.similarity(name_to_id(user1), name_to_id(user3))

            st.write(f"{user1} similarity to {user2} is {simil1:.2f}")
            st.write(f"{user1} similarity to {user3} is {simil2:.2f}")

    except:
        st.write("")


if selected_task == "Find Similar for 3 users":
    st.write("Write three Twitter user names (exact names).")
    st.write("we will find the most matching results for their average")

    c1, c2, c3 = st.columns(3)

    user1 = c1.text_input("User1: ", "")
    user2 = c2.text_input("User2:", "")
    user3 = c3.text_input("User3:", "")
    try:
        res = ""
        if st.button("Go"):
            word1 = name_to_id(user1)
            word2 = name_to_id(user2)
            word3 = name_to_id(user3)

            res = sv_model.predict_output_word([word1, word2, word3], topn=10)

            result_df = pd.DataFrame(
                columns=(["User Name", "Name", "Description", "URL", "Wiki"])
            )

            for item in res:
                username = id_to_name(item[0])

                if username != "Series([], )":
                    desc = ud_df[ud_df["screen_name"] == username][
                        "description"
                    ].to_string(index=False)
                    name = ud_df[ud_df["screen_name"] == username]["name"].to_string(
                        index=False
                    )
                    wiki = ud_df[ud_df["screen_name"] == username][
                        "wikipedia"
                    ].to_string(index=False)
                    url = "http://twitter.com/" + username
                    result_df = result_df.append(
                        {
                            "User Name": username,
                            "Name": name,
                            "Description": desc,
                            "URL": url,
                            "Wiki": wiki,
                        },
                        ignore_index=True,
                    )

            # link is the column with hyperlinks
            result_df["URL"] = result_df["URL"].apply(make_clickable)
            result_df["Wiki"] = result_df["Wiki"].apply(make_clickable)
            result_df_html = result_df.to_html(escape=False)
            st.write(result_df_html, unsafe_allow_html=True)

    except:
        st.write("")
