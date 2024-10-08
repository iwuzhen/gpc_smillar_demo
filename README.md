# streamlit-searchbox 全集刷新bug
pip install git+https://github.com/salmanrazzaq-94/streamlit-searchbox.git@seachbox_compaitibility_enhancement


## sub module

```
# add submodule
git submodule add git@github.com:iwuzhen/page-gutenberg-vs-reddit-via-streamlit.git ./subpage/page-gutenberg-vs-reddit-via-streamlit
```

## fix poetry stuck

```
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
pyenv shell system followed by python3 -m keyring --disable
```