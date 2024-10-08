import streamlit as st

# st.set_page_config(page_title="gpc sentence relavent", layout="wide")

create_page = st.Page("./gpc_demo/gpc_demo.py", title="sentence relavent", icon=":material/edit:",)
summarize_page = st.Page("./gpc_demo/gpc_summarize_demo.py", title="summarize", icon=":material/edit:",)
summarize_with_tree_page = st.Page("./gpc_demo/gpc_summarize_demo_with_tree_v1.py", title="summarize with tree similarity", icon=":material/edit:",)
gutenberg_vs_reddit_page = st.Page("./subpage/page-gutenberg-vs-reddit-via-streamlit/index.py", title="gutenberg vs reddit", icon=":material/edit:",)



pg = st.navigation([create_page, summarize_page, summarize_with_tree_page, gutenberg_vs_reddit_page])
st.set_page_config(page_title="gpc smillar demo", page_icon=":material/edit:",  layout="wide")
pg.run()