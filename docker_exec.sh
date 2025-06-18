docker run -d --gpus all -it --rm \
	--ipc=host \
	--network=host \
	--shm-size=8g \
	-v $(pwd)/shap-e:/workspace/shap-e \
	--name ai_system_container \
	ai_system
