# Chameleon: Robust Website Fingerprinting Defense via Many-to-Many Traffic Morphing
To support the latest graphic devices, we tested the code under both Python 3.10+. The conda environment I provided is based on Python 3.10 with PyTorch 2.8 and CUDA 12.8.


## How to use
### Install dependencies

```bash
conda env create -f py310.yml
conda activate py310
```

### Dataset Format
It has been a convention to name a trace as "A-B.cell" or "C.cell". 
Here, A is the class number of the trace, and B is the instance number of the trace. 
"C.cell" is the C-th non-monitored trace in the dataset.

### Train and Test an Attack

#### Closed World
```bash
python run_attack.py  --attack df --dataset DF
```
#### Open World
```bash
python run_attack.py --attack df --dataset DF --open-world
```
This command will perform a 10-cross-validation attack on the given dataset. 

``--attack`` specifies the attack to evaluate (currently supports DF and Tik-Tok).

``--dataset`` specifies the used dataset.

``--open-world`` makes an open-world evaluation (by default: closed-world evaluation)

``--one-fold`` only run one fold instead of 10 cross-validation.

``--suffix`` specifies the suffix of each file in the dataset (By default: `.cell`). 
Change it if your dataset does not end with `.cell`.

### Simulate a Defense  (If the direction is only considered for the defense method, please change the input dataset dimension for the attack model.)
#### Non-adversarial perturbation works
```bash
Closed World:$ python run_defense.py --defense chameleon --config-path ./defenses/config/chameleon.ini --dataset DF

Open World:$ python run_defense.py --defense chameleon --config-path ./defenses/config/chameleon.ini --dataset DF --open-world
```
#### Adversarial perturbation works
```bash
Closed World:$ python run_defense.py --defense gapdis --config-path ./defenses/config/gapdis.ini --dataset DF --attack df

Open World:$ python run_defense.py --defense gapdis --config-path ./defenses/config/gapdis.ini --dataset DF -- attack df --open-world
```

### Simulate WF defense without adversarial training
To run WF defense without adversarial training, first run Closed- or Open world training to get training models.
1. Closed-world:
```bash
python run_defense_without_adv.py --defense chamemeon --dataset DF --attack df
```
2. Open-world:
```bash
python run_defense_without_adv.py --defense chamemeon --dataset DF --attack df --open-world
```

> [!WARNING]
> For some adversarial perturbation methods (Minipath, GAPDiS, etc.), users need to run the attack model to get the training model first, then run the defense model with the proper attack model loading. The original saved training model has a packet length at the end; the user needs to remove the length and save it to a new file. For example, save DF_df_5000.h5 to DF_df.h5.  

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
It will generate a `defconn_bridgeline.txt` in `/var/lib/tor/chameleon/pt_state`, containing a certification used for handshake as well as the configured parameters.

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

2. **DS-19 Dataset**
   - Top 100 websites, 100 traces each
   - Paper: [Zero-delay Lightweight Defenses against Website Fingerprinting](https://www.usenix.org/system/files/sec20-gong.pdf)

We thank the authors for making these datasets publicly available.
