copy_env:
	cp -r ./.streamlit ~/.streamlit

demo:
	poetry run streamlit run gpc_demo/gpc_demo.py --server.port 18502

main:
	poetry run streamlit run main.py --server.port 18502

sync:
	rsync -avz --delete --exclude=".git" ./* 192.168.1.221:~/workspace/gpc_demo