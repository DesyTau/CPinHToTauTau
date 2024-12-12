A Columnflow based analysis framework from IPHC and DESY

<!-- marker-before-logo -->

<div style="text-align: center;">
    <img src="assets/logo.png" alt="Logo" style="width: 400px; height: 220px; display: block; margin: 0 auto;">
</div>

<!-- marker-after-logo -->

### Resources

- [columnflow](https://github.com/columnflow/columnflow/tree/master)
- [law](https://github.com/riga/law)
- [order](https://github.com/riga/order)
- [luigi](https://github.com/spotify/luigi)

### Setting up the framework (on the main branch)
Clone from the main github repo:
```
git clone --recurse-submodules git@github.com:DesyTau/CPinHToTauTau.git
```
ATTENTION: If you are setting up the framework on lxplus (i.e. on you /afs/cern.ch/user/user_first_char/cern_user_name/ you will encounter problems with micromaba.
A work around to this is selecting the /eos/user/user_first_char/cern_user_name/software as the path for the vens, cmssw and conda folders. As far as I know only afs is giving issues, columnflow people are aware of this but they can not do anything (Jacopo 11/12/24)
This can be achieved by specifying the path that you want to use in the iterative setting up procedure 

With this is mind, now you can set up the framework by:
```
cd CPinHToTauTau
source setup.sh your_setup_name
```
The first time this can take ~10min, but then you are good to go! 
### Setting up the framework used at DESY
Before the previous last step, there are few extra things to do.

But lets do the things step by step starting for cloning the repository:
REMEMBER: if you are planning to develop the code, forking the repo and then cloninf the fork is mandatory!
```
git clone --recurse-submodules git@github.com:DesyTau/CPinHToTauTau.git
cd CPinHToTauTau
```
Then you need to use the following setup: https://github.com/DesyTau/CPinHToTauTau/blob/main/.gitmodules

This mean that essentially these 3 additional steps need to be performed:
1) CPinHToTauTau: https://github.com/DesyTau/CPinHToTauTau/tree/desy_dev; branch desy_dev
```
git checkout origin/desy_dev
```
This is enough to get the core of the analysis scripts and to have the modules/jsonpog-integration repository.
This is necessary for any kind of correction based on file.json

2) cmsdb: https://github.com/DesyTau/cmsdb/tree/desy_dev; branch desy_dev 
```
cd modules/cmsdb
```
```
git checkout origin/desy_dev
```
This is necessary for the definition of campaigns, processes and datasets.

3) columnflow: https://github.com/DesyTau/columnflow; branch production 
```
cd ../columnflow/
```
```
git checkout origin/production
```
This is necessary to have a stable version of columnflow that it is known to work with the current analysis set up.

Now you are finally ready to set up the framework!
```
cd ~
cd CPinHToTauTau
```
ATTENTION: If you are setting up the framework on lxplus (or on any other afs path) see above.
```
source setup.sh your_set_up_name
```
Enjoy the potential of columnflow and do not hesitate to reach us for any issue you may encounter!


