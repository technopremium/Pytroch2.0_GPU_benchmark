# Use the specified NVIDIA PyTorch container as the base image
FROM nvcr.io/nvidia/pytorch:23.04-py3

# Clone the Pytroch2.0_GPU_benchmark repository
RUN git clone https://github.com/technopremium/Pytroch2.0_GPU_benchmark.git /workspace/Pytroch2.0_GPU_benchmark

# Set the working directory to the cloned repository
WORKDIR /workspace/Pytroch2.0_GPU_benchmark

# Install the requirements from the requirements.txt file
RUN pip install -r requirements.txt

# Run the training job using the trainer.py script
CMD ["python3", "trainer.py"]
