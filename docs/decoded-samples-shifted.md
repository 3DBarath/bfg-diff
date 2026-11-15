# Shifted BFG-Diff Step 1500 (`last.pt`) Decoded Samples

Side-by-side completions across 5 standard prompts comparing:
1. **Causal Generate:** native causal autoregression (`model.generate(do_sample=False)`)
2. **C=32, k=8 Single Block:** 4 tok/pass non-autoregressive canvas
3. **C_sub=8, k_sub=4 Parallel:** 2 tok/pass sliding sub-blocks with geometric carrier-island frontier unmasking
4. **C_sub=8, k_sub=8 Bound:** 1 tok/pass sliding sub-blocks (causal-diffusion bound)

---

### Prompt 0: `'Once upon a time, in a small coastal village, the fishermen noticed that'`
- **causal generate:**
  > `' the sea was becoming more and more polluted. The fish were dying, and the sea was becoming less and less colorful. The fishermen realized that something was wrong,'`
- **C=32, k=8 single block:**
  > `' the fish of their the sea were not as abundant the fish the fish the water. They decided to take the fish the water the water the water the fish the'`
- **C_sub=8, k_sub=4 parallel:**
  > `' the fisherman who was the to the most successful was the to the most fisherman. The fishermen were that to that the, that the most the fish the'`
- **C_sub=8, k_sub=8 bound:**
  > `' the fish caught in the were in the shape of a regular. They they the shape of a regular. triangle, and they decided to make to make a.'`

---

### Prompt 1: `'To configure the proxy, edit the settings file and set'`
- **causal generate:**
  > `' the proxy server address and port. The proxy server address is the IP address of the server that you want to use for proxying. The proxy server port is'`
- **C=32, k=8 single block:**
  > `' the value proxy proxy=1\nThe proxy is proxy is used the proxy\nThe proxy is the proxy the proxy is used\n proxy is used\nThe proxy'`
- **C_sub=8, k_sub=4 parallel:**
  > `' the proxy server address the\n proxy server port and the proxy server proxy protocol.\nThe proxy server protocol is to the proxy server protocol. The\n is\nproxy'`
- **C_sub=8, k_sub=8 bound:**
  > `' the proxy server address the. proxy server port, and proxy server port..\n\nThe proxy server is should the to the proxy server address and port the to the'`

---

### Prompt 2: `'## Quick start\n\nInstall the package, then run'`
- **causal generate:**
  > `' the following command:\n\n```\npython3 -m pip install -r requirements.txt\n```\n\n## Installation\n\n### Requirements\n\nThe following requirements are required to run'`
- **C=32, k=8 single block:**
  > `' the following command to install the package\n\n```\npip install -py\n\n``\n\n\n\n##\n\n\n\n\n\n## Installation\n\n### 1.\n\nInstall the'`
- **C_sub=8, k_sub=4 parallel:**
  > `' the following command to the to\n\n```\npython -m -.py\n```\n\nThis will run the a to the command line, and the python to the'`
- **C_sub=8, k_sub=8 bound:**
  > `' the following command to the to\n\n```\npython3.3.py\n```\n\nThis will run the a to the command line, and the python will the'`

---

### Prompt 3: `'{\n  "task": "summarise",\n  "input": "Quarterly revenue rose'`
- **causal generate:**
  > `' by 10%",\n  "output": "Quarterly revenue rose by 10%",\n  "output": "Quarterly revenue rose by 1'`
- **C=32, k=8 single block:**
  > `' by 10% in",\n",\n",\n  "":": "sum",\n  "": "summarise",\n ",\n ",\n  "": "'`
- **C_sub=8, k_sub=4 parallel:**
  > `' by 10%",\n  "output": "Quarter",\n 1",\n  "output": "",\n1",\n  "output": "1",\n '`
- **C_sub=8, k_sub=8 bound:**
  > `' by 10%",\n  "output": "Quarter",\n  "",\n  "output": "",\n1",\n  "task": "sum",\n '`

---

### Prompt 4: `'- first, collect the measurements\n- second, clean the outliers\n- third,'`
- **causal generate:**
  > `' remove outliers\n- fourth, plot the data\n- fifth, plot the data\n- sixth, plot the data\n- seventh, plot the data\n-'`
- **C=32, k=8 single block:**
  > `' plot the data\n-\n- fourth, plot,, and finally, save the plot\n-\n\n- fifth and, save the plot\n-\n'`
- **C_sub=8, k_sub=4 parallel:**
  > `' plot the data\n- fourth\n, plot the data\n the\n- fifth, plot the data\n- sixth, plot the data\n-\n- seventh'`
- **C_sub=8, k_sub=8 bound:**
  > `' plot the data\n- fourth\n, plot the data\n the\n- fifth, plot the data\n- sixth, plot the data\n-\n- seventh'`
