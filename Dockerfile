

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3 python3-pip

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# ✅ Install PyTorch manually from PyTorch official repository
RUN pip3 install torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Copy the rest of the code
COPY . ./

# Run the script
CMD ["python3", "main.py"]
