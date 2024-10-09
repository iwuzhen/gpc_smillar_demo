copy_env:
	cp -r ./.streamlit ~/.streamlit

demo:
	poetry run streamlit run gpc_demo/gpc_demo.py --server.port 18502

main:
	poetry run streamlit run main.py --server.port 18502

build:
	docker build -t test .