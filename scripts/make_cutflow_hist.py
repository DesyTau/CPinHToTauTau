# . $CF_BASE/sandboxes/venv_columnar_dev.sh
import matplotlib.pyplot as plt
import mplhep
import pickle
import numpy as np
import sys


def create_cutflow_histogram(cuts, data, xlabel="Selections", ylabel="Selection efficiency", title="", log=False, rel=False, save_path=None):
    """
    Create a cutflow histogram using Matplotlib with CMS style and save it to a PDF file.

    Parameters:
    - cuts: List of strings representing the names of the cuts.
    - data: List of integers representing the corresponding event counts for each cut.
    - xlabel: Label for the x-axis (default is "Cuts").
    - ylabel: Label for the y-axis (default is "Events").
    - title: Title of the plot (default is "Cutflow Histogram").
    - save_path: Path to save the PDF file. If None, the plot will be displayed but not saved.

    Returns:
    - fig: Matplotlib figure object.
    - ax: Matplotlib axis object.
    """

    # Set CMS style
    plt.style.use(mplhep.style.CMS)
   
    # Create Matplotlib figure and axis
    fig, ax = plt.subplots()
    if log: plt.yscale('log')
    # Plotting the cutflow histogram
    color = ['blue', 'black','red']
    
    for i, (name, n_evt) in enumerate(data.items()):
        
        n_evt = np.array(n_evt)
        for cut_name, the_n_evt in zip(cuts,n_evt):
            print(f"{cut_name}: {the_n_evt}")
        if rel:
            x = cuts[1:]
            y = n_evt[1:]/n_evt[:-1]
        elif not log:
            x = cuts
            y = n_evt/n_evt[0]
        else:
            x = cuts
            y = n_evt
        print(f'Event numbers:')

        #ax.step(x, y , color=color[i], where='mid',marker='o', linewidth=1.2, alpha=0.8, label=r"Z $\rightarrow \ell \ell$")
        #ax.step(x, y , color=color[i], where='mid',marker='o', linewidth=1.2, alpha=0.8, label=r"W $\rightarrow \ell \nu$")
        ax.step(x, y , color=color[i], where='mid',marker='o', linewidth=1.2, alpha=0.8, label=r"EGamma_2022C Data")
        # ax.step(x, y , color=color[i], where='mid',marker='o', linewidth=1.2, alpha=0.8, label=r"EGamma_2022D Data")
        # ax.step(x, y , color=color[i], where='mid',marker='o', linewidth=1.2, alpha=0.8, label=r"Muon_2022C Data")
        # ax.step(x, y , color=color[i], where='mid',marker='o', linewidth=1.2, alpha=0.8, label=r"Muon_2022D Data")
        # ax.step(x, y , color=color[i], where='mid',marker='o', linewidth=1.2, alpha=0.8, label=r"Tau_2022C Data")
        # ax.step(x, y , color=color[i], where='mid',marker='o', linewidth=1.2, alpha=0.8, label=r"Tau_2022D Data")
        #ax.step(x, y , color=color[i], where='mid',marker='o', linewidth=1.2, alpha=0.8, label=r"$H_{ggf} \rightarrow \tau\tau$")        

        for i, txt in enumerate(y):
            the_txt = f'{txt:.4f}' if rel else f'{txt:.0f}'
            ax.annotate(the_txt, (x[i], y[i]),
                        textcoords="offset points",
                        xytext=(0,-20),
                        ha='center',
                        fontsize=10)

    if log:
        if rel: ax.set_ylim((1*10**-6,2*10**0))
        else: 
            pow_nevt = int(np.log10(n_evt[0]))+1.1
            ax.set_ylim((10**(2),10**pow_nevt))
    
    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticklabels(cuts[1:] if rel else cuts, rotation=45, ha='right')
    
    # Add legend
    ax.legend()
    ax.grid(True, which='both', axis='y')
    ax.grid(True, which='major', axis='x')
    label_options = {
    "wip": "Work in progress",
    "pre": "Preliminary",
    "pw": "Private work",
    "sim": "Simulation",
    "simwip": "Simulation work in progress",
    "simpre": "Simulation preliminary",
    "simpw": "Simulation private work",
    "od": "OpenData",
    "odwip": "OpenData work in progress",
    "odpw": "OpenData private work",
    "public": "",
    }
    cms_label_kwargs = {
        "ax": ax,
        "llabel": label_options.get("simwip"),
        "fontsize": 22,
        "data": True,
        'rlabel': name
    }
    mplhep.cms.label(**cms_label_kwargs)
   
    plt.tight_layout()
    
     # Save to PDF if save_path is provided
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        # Show the plot if save_path is not provided
        plt.show()


    return fig, ax

def get_hist_values(pickle_file):
    file_obj = open(pickle_file, 'rb')
    data = pickle.load(file_obj)
    hist = data.profile(axis=0)
    cuts = []
    values = []
    print(hist)
    for cut_name in hist.axes[1]:
        cuts.append(f'{cut_name}')
        values.append(hist[0,f'{cut_name}',0,0].count)
    return cuts, values

#DY
#path22 = '/afs/cern.ch/user/j/jmalvaso/CPinHToTauTau/data/cf_store/analysis_httcp/cf.CreateCutflowHistograms/run3_2022_preEE/dy_incl/nominal/calib__main/sel__main__steps_trigger_met_filter_b_veto_dilepton_veto_selected_hcand_selected_hcand_trigmatch_single_hcand_extra_lepton_veto_decay_prods_are_ok/dev/cutflow_hist__event.pickle'
# #WJ
#path22 = '/afs/cern.ch/user/j/jmalvaso/CPinHToTauTau/data/cf_store/analysis_httcp/cf.CreateCutflowHistograms/run3_2022_preEE/wj_incl/nominal/calib__main/sel__main__steps_trigger_met_filter_b_veto_dilepton_veto_selected_hcand_selected_hcand_trigmatch_single_hcand_extra_lepton_veto_decay_prods_are_ok/dev/cutflow_hist__event.pickle'

# #Data
# #mu_C
#path22 = '/afs/cern.ch/user/j/jmalvaso/CPinHToTauTau/data/cf_store/analysis_httcp/cf.CreateCutflowHistograms/run3_2022_preEE/data_mu_C/nominal/calib__main/sel__main__steps_trigger_met_filter_b_veto_dilepton_veto_selected_hcand_selected_hcand_trigmatch_single_hcand_extra_lepton_veto_decay_prods_are_ok/dev/cutflow_hist__event.pickle'
# #mu_D
#path22 = '/afs/cern.ch/user/j/jmalvaso/CPinHToTauTau/data/cf_store/analysis_httcp/cf.CreateCutflowHistograms/run3_2022_preEE/data_mu_D/nominal/calib__main/sel__main__steps_trigger_met_filter_b_veto_dilepton_veto_selected_hcand_selected_hcand_trigmatch_single_hcand_extra_lepton_veto_decay_prods_are_ok/dev/cutflow_hist__event.pickle'
# #e_C
# path22 = '/afs/cern.ch/user/j/jmalvaso/CPinHToTauTau/data/cf_store/analysis_httcp/cf.CreateCutflowHistograms/run3_2022_preEE/data_e_C/nominal/calib__main/sel__main__steps_trigger_met_filter_b_veto_dilepton_veto_selected_hcand_selected_hcand_trigmatch_single_hcand_extra_lepton_veto_decay_prods_are_ok/dev/cutflow_hist__event.pickle'
# #e_D
#path22 = '/afs/cern.ch/user/j/jmalvaso/CPinHToTauTau/data/cf_store/analysis_httcp/cf.CreateCutflowHistograms/run3_2022_preEE/data_e_D/nominal/calib__main/sel__main__steps_trigger_met_filter_b_veto_dilepton_veto_selected_hcand_selected_hcand_trigmatch_single_hcand_extra_lepton_veto_decay_prods_are_ok/dev/cutflow_hist__event.pickle'
# #tau_C
#path22 = '/afs/cern.ch/user/j/jmalvaso/CPinHToTauTau/data/cf_store/analysis_httcp/cf.CreateCutflowHistograms/run3_2022_preEE/data_tau_C/nominal/calib__main/sel__main__steps_trigger_met_filter_b_veto_dilepton_veto_selected_hcand_selected_hcand_trigmatch_single_hcand_extra_lepton_veto_decay_prods_are_ok/dev/cutflow_hist__event.pickle'

path22 = '/afs/cern.ch/user/j/jmalvaso/CPinHToTauTau/data/cf_store/analysis_MSSM_H_tt/cf.CreateCutflowHistograms/run3_2022_preEE_etau/data_e_C/nominal/calib__main/sel__main__steps_trigger_met_filter_has_at_least_2_leptons_selected_hcand_selected_hcand_trigmatch_single_hcand_single_lepton_veto_second_lepton_veto_nans_removed_jet_veto_map/desy_dev/cutflow_hist__event.pickle'
cuts, values22 = get_hist_values(path22)

create_cutflow_histogram(cuts, 
                         data={"2022 preEE": values22},
                         save_path="cutflow_histogram_2022_preEE_DATA_C.png",
                         ylabel="N evt",
                         log=True,
                         rel=False)
