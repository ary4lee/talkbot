TALKBOT, YOUR TALKING AI ASSISTANT!

Talkbot is program that utilizes ASR and local LLMs to create an AI assitant that you can talk to. 

-----------------------------------------------------------------------------------
SETUP:

Conda (Windows):
REQUIRES Desktop Developement with C++ (Install from the Visual Studio Installer, which can be downloaded at https://visualstudio.microsoft.com/downloads/?q=build+tools)	


- Download the Llama GGUF Model at https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
- Create a new miniconda environment and install Python 3.12:

conda create --name [environment name] python==3.12

- Install the requirements.txt file

pip install -r requirements.txt

- Setup your development program to use your conda environment as the interpreter
- Open talkbot.py
- Change to model path to where the Llama GGUF model from eariler is located
- Run the program!
- To use:
	- Say: Hey, talkbot
	- It will say "What question do you have?". Speak back your question, and it will respond.