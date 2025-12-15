#!/bin/bash
docker ps -a --filter "name=llamppl-worker" --format "{{.Names}}" | while read -r name; do
    echo "Stopping and removing container: $name"
    docker stop "$name" > /dev/null 2>&1
    docker rm "$name" > /dev/null 2>&1
done

detect_num_workers() {
    if ! command -v nvidia-smi &> /dev/null
    then
        echo "nvidia-smi could not be found. Defaulting to 1 worker."
        NUM_WORKERS=1
    else
        NUM_WORKERS=$(nvidia-smi -L | wc -l)
        if [ "$NUM_WORKERS" -eq 0 ]; then
            echo "No GPUs detected by nvidia-smi. Defaulting to 1 worker."
            NUM_WORKERS=1
        else
            echo "Detected ${NUM_WORKERS} GPUs. Will launch that many workers."
        fi
    fi
    if [ "$NUM_WORKERS" -lt 1 ]; then
        NUM_WORKERS=1
        echo "Adjusted NUM_WORKERS to 1 (minimum)."
    fi
    echo "Final NUM_WORKERS set to: ${NUM_WORKERS}"
}

detect_num_workers

BASE_HOST_PORT=8001
LAUNCH_DELAY_SECONDS=120

echo "Starting ${NUM_WORKERS} llamppl inference workers..."

for i in $(seq 0 $((NUM_WORKERS-1))); do
    HOST_PORT=$((BASE_HOST_PORT + i))
    DEVICE_ID=$i
    CONTAINER_NAME="llamppl-worker-${i}"
    RANK=$i

    echo "Launching worker ${i} on physical GPU ${DEVICE_ID} (mapped to container's /dev/cuda:0)..."
    docker run -d \
      --name "${CONTAINER_NAME}" \
      --gpus "device=${DEVICE_ID}" \
      -p "${HOST_PORT}":8000 \
      -e WORKER_ID="${i}" \
      -e CUDA_VISIBLE_DEVICES="0" \
      --shm-size="4g" \
      -v "$(pwd)/.model_cache:/app/huggingface" \
      llamppl-inference-server

    if [ "$i" -lt $((NUM_WORKERS-1)) ]; then
        echo "Waiting for ${LAUNCH_DELAY_SECONDS} seconds before launching next worker..."
        sleep ${LAUNCH_DELAY_SECONDS}
    fi
done

echo "All workers launched. Use 'docker ps' to check status."
echo "You can check individual worker logs with 'docker logs llamppl-worker-X'"