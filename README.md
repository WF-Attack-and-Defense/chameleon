# Chameleon: Robust Website Fingerprinting Defense via Many-to-Many Traffic Morphing
To support the latest graphics devices, we tested the code under Python 3.10.

:warning: :warning: :warning: This software is a research prototype intended solely for experimental and evaluation purposes. :warning: :warning: :warning:

## How to use
### Install dependencies
1. Python
```bash
pip install -r requirements.txt
```
2. Conda
```bash
conda env create -f py310.yml
conda activate py310
```

### Dataset Format
It has been a convention to name a trace as "A-B.cell" or "C.cell". 
A is the class number of the trace, and B is the instance number of the trace. 
"C.cell" is the C-th non-monitored trace in the dataset.

Extract sample dataset
```
tar -xJf ds-19.tar.xz
```

#### Defended dataset generation
```bash
Closed World:$ python run_defense.py --defense chameleon --config-path ./defenses/config/chameleon.ini --dataset DF

Open World:$ python run_defense.py --defense chameleon --config-path ./defenses/config/chameleon.ini --dataset DF --open-world
```

## Source tree

```
datasets/
├── sample.zip/                      # Sample dataset: 100 x 100 traces
src/
├── attacks/                         # Website-fingerprinting attack models
│   ├── __init__.py                  
│   ├── base.py                      # Shared Attack class: data loading, training, CV, checkpoints
│   ├── df.py                        # Deep Fingerprinting attack (packet-direction CNN)
│   ├── netclr.py                    # NetCLR attack: SimCLR pre-train then supervised fine-tune
│   ├── rf.py                        # Robust Fingerprinting attack (TAM features + CNN)
│   ├── var_cnn.py                   # Var-CNN attack (timing + direction residual CNN)
│   ├── config/
│   │   └── const.py                 # NetCLR SimCLR pre-training hyperparameters
│   └── modules/
│       ├── __init__.py             
│       ├── df.py                    # DF 1D CNN classifier 
│       ├── netclr.py                # NetCLR backbone, SimCLR head, and pre-training loop
│       ├── rf.py                    # RF 1D CNN classifier
│       └── var_cnn.py               # Var-CNN dilated residual network
├── defenses/
│   ├── __init__.py                  
│   ├── base.py                      
│   ├── chameleon.py                 # Chameleon: select traces, build a radix trie, morph traffic
│   └── config/
│       ├── __init__.py              
│       ├── chameleon.ini            # Chameleon parameters (selection, trie, mutation)
│       └── config.py                # INI loader and ChameleonConfig field types
├── utils/
│   ├── __init__.py                 
│   ├── checkpoint.py                # Save, load, and clean up attack model checkpoints
│   ├── data.py                      # PyTorch Dataset that extracts features from trace files
│   ├── general.py                   # Trace I/O, feature transforms, seeds, file lists
│   ├── logger.py                    # Console (and optional file) logging setup
│   ├── metric.py                    # Closed/open-world WF metrics (TP/FP) for Ignite
│   ├── netclr_augment.py            # Burst-based SimCLR augmentations for NetCLR
│   ├── overhead-time-data.py        # CLI: bandwidth/time overhead of a defended dataset
│   ├── perturb_util.py              # Load an attack and evaluate it on defended traces
│   └── chameleon/
│       ├── feature_extract.py       # Packet-level statistical features used in selection
│       ├── predataprocessing.py     # Trace selection and NCC grouping before morphing
│       └── radixTrie.py             # Direction-prefix trie used to pick morphing targets
├── run_attack.py                    # Train/evaluate attacks (k-fold CV; optional -d full-set train)
├── run_defense.py                   # Generate a Chameleon-defended dataset
└── run_defense_evaluation.py        # Run a trained attack against Chameleon-defended traces (Without adversarial training)
```

## Pluggable Transport Deployment

### 1. Build Obfs4proxy
```bash
go build -o obfs4proxy/obfs4proxy ./obfs4proxy
```
The compiled binary at `./PluggableTransport/obfs4proxy/obfs4proxy`
### 2. Move the PluggableTransport and JSON file to the `/usr/bin` folder
```bash
sudo cp ./PluggableTransport/obfs4proxy/obfs4proxy /usr/bin/obfs4proxy-chameleon
sudo chown root:root /usr/bin/obfs4proxy-chameleon
sudo chmod 755 /usr/bin/obfs4proxy-chameleon

sudo cp ./ds-19.json /usr/bin/
sudo chown root:root /usr/bin/ds-19.json
sudo chmod 755 /usr/bin/ds-19.json

sudo sh -c 'cat > /etc/apparmor.d/local/system_tor <<EOF
# allow chameleon PT binary and dataset
/usr/bin/obfs4proxy-chameleon rix,
/usr/bin/ds-19.json r,
EOF'
sudo apparmor_parser -r /etc/apparmor.d/system_tor
sudo systemctl restart tor@default
```

You can use the following command to check whether the PluggableTransport is running successfully:
```bash
sudo journalctl -k -n 50 --no-pager | grep -Ei "system_tor|ds-19|obfs4proxy|denied"
sudo journalctl -u tor@default -n 80 -l --no-pager | grep -Ei "managed proxy|method error|didn't launch|chameleon"
```

### 3. Configure the Bridge
Open `/etc/tor/torrc`, then add the following configuration at the end of the configuration file:
```
DataDirectory /var/lib/tor/chameleon
Log notice stdout
SOCKSPort 9052

BridgeRelay 1
PublishServerDescriptor 0
ORPort auto
ExtORPort auto
ExitPolicy reject *:*
Nickname chameleon

ServerTransportListenAddr chameleon 0.0.0.0:34000
ServerTransportPlugin chameleon exec /usr/bin/obfs4proxy-chameleon
```
It will generate a `defconn_bridgeline.txt` in `/var/lib/tor/chameleon/pt_state`, containing a certificate used for the handshake as well as the configured parameters.

### 4. Configure the client
The client's torrc file is like:
```
DataDirectory /var/lib/tor/chameleon
Log notice stdout    
SOCKSPort 9050
ControlPort 9051
UseBridges 1
Bridge chameleon xxx.xxx.xxx.xx:34000 cert=<cert>
ClientTransportPlugin wfgan exec /usr/bin/obfs4proxy-chameleon
```
`xxx.xxx.xxx.xx` is the bridge IP address. You can get `<cert>` from the `defconn_bridgeline.txt` file on the Bridge.

## Datasets
This implementation uses two public website fingerprinting datasets:

1. **Sirinam et al. (Tik-Tok) Dataset**
   - 95 websites, 1,000 traces each
   - Paper: [Deep Fingerprinting: Undermining Website Fingerprinting Defenses with Deep Learning](https://dl.acm.org/doi/pdf/10.1145/3243734.3243768)

2. **ds-19 Dataset**
   - Top 100 websites, 100 traces each
   - Paper: [Zero-delay Lightweight Defenses against Website Fingerprinting](https://www.usenix.org/system/files/sec20-gong.pdf)

2. **GTT23 Dataset**
   - Top 100 websites, 2000 traces each
   - Paper: [A Measurement of Genuine Tor Traces for Realistic Website Fingerprinting](https://link.springer.com/chapter/10.1007/978-3-032-18268-5_13)

## Attack Models
The defense model evaluation uses WF attack models from our project and Website-Fingerprinting-Library (WFLIB):
   - Paper: [Robust and Reliable Early-Stage Website Fingerprinting Attacks via Spatial-Temporal Distribution Analysis](https://dl.acm.org/doi/epdf/10.1145/3658644.3670272)
   - Github: [https://github.com/FIND-Lab/Website-Fingerprinting-Library](https://github.com/FIND-Lab/Website-Fingerprinting-Library)

We thank the authors for making these datasets and WFLIB publicly available.
