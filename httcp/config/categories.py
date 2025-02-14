# coding: utf-8

"""
Definition of categories.
"""

import order as od

from columnflow.config_util import add_category
from columnflow.util import DotDict
from columnflow.util import maybe_import
np = maybe_import("numpy")


def add_categories(config: od.Config,
                   channel = None) -> None:
    
    
    def add_base_categories(config, channel, category_map, base_selection=[]):
        base_cat = config.get_category('_'.join(('cat',channel)))      
        for i, (cat_name, cat) in enumerate(category_map.items()):
            kwargs = {
                'name'      : '_'.join((base_cat.name, cat_name)),
                'selection' : base_selection + cat.selection,
                'id'        : 100*(i+1)+ base_cat.id,
                'label'     :' '.join((base_cat.label.split(' ')[0], cat.label if 'label' in cat else cat_name))
            }
            add_category(config, **kwargs)
            
    
    def add_child_categories(config, parent_categories, child_category_map):
        max_child_id = int(np.max(config.categories.ids())/10000)
        for cat_name in parent_categories:
            if cat_name in ['incl', 'cat_mutau', 'cat_etau']: continue
            parent_cat = config.get_category(cat_name)
            #in case of add_child_categories function was applied previosly its needed to provide unique ids, 
            # max_child_id will be added to iterator
            for i, (child_name, child_cat) in enumerate(child_category_map.items()):
                kwargs = {
                    'name'      : '__'.join((parent_cat.name, child_name)),
                    'selection' : parent_cat.selection + child_cat.selection,
                    'id'        : 10000*(i+1+max_child_id)+ parent_cat.id,
                    'label'     : ' '.join((parent_cat.label, child_cat.label if 'label' in child_cat else child_name))
                }
                if 'signal_reg' in parent_cat.aux:
                    kwargs['aux']['signal_reg'] = True
                add_category(config, **kwargs)
    
    """
    Adds all categories to a *config*.
    ids from 1 to 9 are reserved for channels
    """
    
    add_category(
        config,
        name="incl",
        id=1,
        selection=["cat_incl"],
        label="inclusive",
    )
    ##############################################
    ### Main categories for the three channels ###
    ##############################################
    
    if channel=='mutau':
        add_category(
            config,
            name="cat_mutau",
            id=2,
            selection=["cat_mutau"],
            label=r"$\mu\tau$ inclusive",
        )

    elif channel=='etau':
        add_category(
            config,
            name="cat_etau",
            id=3,
            selection=["cat_etau"],
            label=r"$e\tau$ inclusive",
        )
    
    #Define initial category map with selections and call the function
    #Don't change this part: it is important for fake factor method    
    base_selection = [f'cat_{channel}', 'os_charge']
    
    category_map  = DotDict.wrap({
        "sr"            : { 'selection' : ["mt_cut",      "deep_tau_wp",      "lep_iso"],
                            'label'     : r"signal region",
                            'aux'       : {'is_signal_reg' : True}},
        
        "no_mt"         : { 'selection' : ["deep_tau_wp",      "lep_iso"],
                            'label'     : r"no mT",},
        "ar_wj"         : {'selection' : ["mt_cut",      "deep_tau_inv_wp",  "lep_iso"],},
        "dr_num_wj"     : {'selection' : ["mt_inv_cut",  "deep_tau_wp",      "lep_iso", "tau_no_fakes"],},
        "dr_den_wj"     : {'selection' : ["mt_inv_cut",  "deep_tau_inv_wp",  "lep_iso", "tau_no_fakes"],},
        
        "ar_qcd"        : {'selection' : ["mt_cut",      "deep_tau_inv_wp",  "lep_iso"],},
        "dr_num_qcd"    : {'selection' : ["mt_cut",      "deep_tau_wp",      "lep_inv_iso", "tau_no_fakes"],},
        "dr_den_qcd"    : {'selection' : ["mt_cut",      "deep_tau_inv_wp",  "lep_inv_iso", "tau_no_fakes"],},
        
        "ar_yields"     : {'selection' : ["mt_cut",      "deep_tau_inv_wp",  "lep_iso"],},
        })

    add_base_categories(config, channel, category_map, base_selection)
    
    #Add child categories to base categories
    child_category_map  = DotDict.wrap({
        "tau_barrel"    : {'selection' : ["tau_barrel"],
                            'label'     : r"$\eta_{\tau} \leq 1.2$",},
        "tau_endcap"    : {'selection' : ["tau_endcap"],
                            'label'     : r"$\eta_{\tau} > 1.2$",},
        })
     
    # add_child_categories(config,
    #                      parent_categories=config.categories.names(),
    #                      child_category_map=child_category_map)
    
# Old categories in case if someone needs 
# etau_control_reg_barrel_tau = add_category(
#     config,
#     name="etau_control_reg_barrel_tau",
#     id=450 + etau.id,
#     selection=["cat_etau"  ,
#                 "ss_charge"  ,
#                 "mt_cut"     ,
#                 "deep_tau_wp",
#                 "b_veto"     ,
#                 "tau_barrel" ,
#                 ],
#     label=r"$e\tau$ control region\n $\eta_{\tau} \leq 1.2$",
# )

# etau_signal_reg_endcap_tau_no_mt = add_category(
#     config,
#     name="etau_signal_reg_endcap_tau_no_mt",
#     id=500 + etau.id,
#     selection=["cat_etau"   ,
#                "os_charge"  ,
#                "deep_tau_wp",
#                "b_veto"     ,
#                "tau_endcap" ,
#                ],
#     label=r"$e\tau$ no mt signal region\n $\eta_{\tau} > 1.2$",
#     aux={'control_reg': "etau_control_reg_endcap_tau_no_mt"}
# )
# etau_control_reg_endcap_tau_no_mt = add_category(
#     config,
#     name="etau_control_reg_endcap_tau_no_mt",
#     id=550 + etau.id,
#     selection=["cat_etau"   ,
#                "ss_charge"  ,
#                "deep_tau_wp",
#                "b_veto"     ,
#                "tau_endcap" ,
#                ],
#     label=r"$e\tau$ no mt control region\n $\eta_{\tau} > 1.2$",
# )

# etau_signal_reg_barrel_tau_no_mt = add_category(
#     config,
#     name="etau_signal_reg_barrel_tau_no_mt",
#     id=600 + etau.id,
#     selection=["cat_etau"  ,
#                 "os_charge"  ,
#                 "deep_tau_wp",
#                 "b_veto"     ,
#                 "tau_barrel" ,
#                 ],
#     label=r"$e\tau$ no mt signal region\n $\eta_{\tau} \leq 1.2$",
#     aux={'control_reg': "etau_control_reg_barrel_tau_no_mt"}
# )

# etau_control_reg_barrel_tau_no_mt = add_category(
#     config,
#     name="etau_control_reg_barrel_tau_no_mt",
#     id=650 + etau.id,
#     selection=["cat_etau"  ,
#                 "ss_charge"  ,
#                 "deep_tau_wp",
#                 "b_veto"     ,
#                 "tau_barrel" ,
#                 ],
#     label=r"$e\tau$ no mt control region\n $\eta_{\tau} \leq 1.2$",
# )


# etau_signal_reg_no_mt_bveto_wp_mtt = add_category(
#     config,
#     name="etau_signal_reg_no_mt_bveto_wp_mtt",
#     id=700 + etau.id,
#     selection=["cat_etau"  ,
#                "os_charge"  ,
#                "deep_tau_wp_mtt",
#               ],
#     label=r"$e\tau$ signal region no mt, mtt wp, no b veto",
#     aux={'control_reg': "etau_contol_reg_no_mt_bveto_wp_mtt"}
# )

# etau_contol_reg_no_mt_bveto_wp_mtt = add_category(
#     config,
#     name="etau_contol_reg_no_mt_bveto_wp_mtt",
#     id=750 + etau.id,
#     selection=["cat_etau"  ,
#                "ss_charge"  ,
#                "deep_tau_wp_mtt",
#               ],
#     label=r"$e\tau$ control region no mt, mtt wp, no b veto",
# )


# etau_signal_reg_no_mt_bveto_wp_mtt = add_category(
#     config,
#     name="etau_signal_reg_no_mt_bveto",
#     id=800 + etau.id,
#     selection=["cat_etau"  ,
#                "os_charge"  ,
#                "deep_tau_wp",
#               ],
#     label=r"$e\tau$ signal region no mt, no b veto",
#     aux={'control_reg': "etau_contol_reg_no_mt_bveto"}
# )

# etau_contol_reg_no_mt_bveto = add_category(
#     config,
#     name="etau_contol_reg_no_mt_bveto",
#     id=850 + etau.id,
#     selection=["cat_etau"  ,
#                "ss_charge"  ,
#                "deep_tau_wp",
#               ],
#     label=r"$e\tau$ control region no mt, no b veto",
# )

# #################################
# ### mu-tau channel categories ###
# #################################

# mutau_signal_reg = add_category(
#     config,
#     name="mutau_signal_reg",
#     id=100 + mutau.id,
#     selection=["cat_mutau"  ,
#                "os_charge"  ,
#                "mt_cut"     ,
#                "deep_tau_wp",
#                "b_veto"     ,
#               ],
#     label=r"$\mu\tau$ signal region",
#     aux={'control_reg': "mutau_control_reg"}
# )

# mutau_control_reg = add_category(
#     config,
#     name="mutau_control_reg",
#     id=150 + mutau.id,
#     selection=["cat_mutau"  ,
#                "ss_charge"  ,
#                "mt_cut"     ,
#                "deep_tau_wp",
#                "b_veto"     ,
#                ],
#     label=r"$\mu\tau$ control region",
# )
# mutau_signal_reg_no_mt = add_category(
#     config,
#     name="mutau_signal_reg_no_mt",
#     id=200 + mutau.id,
#     selection=["cat_mutau"  ,
#                "os_charge"  ,
#                "deep_tau_wp",
#                "b_veto"     ,
#                ],
#     label=r"$\mu\tau$ no mt",
#     aux={'control_reg': "mutau_control_reg_no_mt"} 
# )
# mutau_control_reg_no_mt = add_category(
#     config,
#     name="mutau_control_reg_no_mt",
#     id=250 + mutau.id,
#     selection=["cat_mutau"  ,
#                "ss_charge"  ,
#                "deep_tau_wp",
#                "b_veto"     ,
#                ],
#     label=r"$\mu\tau$ control region no mt",
# )
# mutau_signal_reg_endcap_tau = add_category(
#     config,
#     name="mutau_signal_reg_endcap_tau",
#     id=300 + mutau.id,
#     selection=["cat_mutau"  ,
#                "os_charge"  ,
#                "mt_cut"     ,
#                "deep_tau_wp",
#                "b_veto"     ,
#                "tau_endcap" ,
#                ],
#     label=r"$\mu\tau$ signal region\n $\eta_{\tau} > 1.2$",
#     aux={'control_reg': "mutau_control_reg_endcap_tau"}
# )
# mutau_control_reg_endcap_tau = add_category(
#     config,
#     name="mutau_control_reg_endcap_tau",
#     id=350 + mutau.id,
#     selection=["cat_mutau"  ,
#                "ss_charge"  ,
#                "mt_cut"     ,
#                "deep_tau_wp",
#                "b_veto"     ,
#                "tau_endcap" ,
#                ],
#     label=r"$\mu\tau$ control region\n $\eta_{\tau} > 1.2$",
# )

# mutau_signal_reg_barrel_tau = add_category(
#     config,
#     name="mutau_signal_reg_barrel_tau",
#     id=400 + mutau.id,
#     selection=["cat_mutau"  ,
#                "os_charge"  ,
#                "mt_cut"     ,
#                "deep_tau_wp",
#                "b_veto"     ,
#                "tau_barrel" ,
#                ],
#     label=r"$\mu\tau$ signal region\n $\eta_{\tau} \leq 1.2$",
#     aux={'control_reg': "mutau_control_reg_barrel_tau"}
# )

# mutau_control_reg_barrel_tau = add_category(
#     config,
#     name="mutau_control_reg_barrel_tau",
#     id=450 + mutau.id,
#     selection=["cat_mutau"  ,
#                "ss_charge"  ,
#                "mt_cut"     ,
#                "deep_tau_wp",
#                "b_veto"     ,
#                "tau_barrel" ,
#                ],
#     label=r"$\mu\tau$ control region\n $\eta_{\tau} \leq 1.2$",
# )

# mutau_signal_reg_endcap_tau_no_mt  = add_category(
#     config,
#     name="mutau_signal_reg_endcap_tau_no_mt",
#     id=500 + mutau.id,
#     selection=["cat_mutau"  ,
#                "os_charge"  ,
#                "deep_tau_wp",
#                "b_veto"     ,
#                "tau_endcap" ,
#                ],
#     label=r"$\mu\tau$ no mt \n $\eta_{\tau} > 1.2$",
#     aux={'control_reg': "mutau_control_reg_endcap_tau_no_mt"}
# )
# mutau_control_reg_endcap_tau_no_mt = add_category(
#     config,
#     name="mutau_control_reg_endcap_tau_no_mt",
#     id=550 + mutau.id,
#     selection=["cat_mutau"  ,
#                "ss_charge"  ,
#                "deep_tau_wp",
#                "b_veto"     ,
#                "tau_endcap" ,
#                ],
#     label=r"$\mu\tau$ no mt control region\n $\eta_{\tau} > 1.2$",
# )

# mutau_signal_reg_barrel_tau_no_mt = add_category(
#     config,
#     name="mutau_signal_reg_barrel_tau_no_mt",
#     id=600 + mutau.id,
#     selection=["cat_mutau"  ,
#                "os_charge"  ,
#                "deep_tau_wp",
#                "b_veto"     ,
#                "tau_barrel" ,
#                ],
#     label=r"$\mu\tau$ no mt \n $\eta_{\tau} \leq 1.2$",
#     aux={'control_reg': "mutau_control_reg_barrel_tau_no_mt"}
# )

# mutau_control_reg_barrel_tau_no_mt = add_category(
#     config,
#     name="mutau_control_reg_barrel_tau_no_mt",
#     id=650 + mutau.id,
#     selection=["cat_mutau"  ,
#                "ss_charge"  ,
#                "deep_tau_wp",
#                "b_veto"     ,
#                "tau_barrel" ,
#                ],
#     label=r"$\mu\tau$ no mt control region\n $\eta_{\tau} \leq 1.2$",
# )

# mutau_signal_reg_no_mt_bveto_wp_mtt = add_category(
#     config,
#     name="mutau_signal_reg_no_mt_bveto_wp_mtt",
#     id=700 + mutau.id,
#     selection=["cat_mutau"  ,
#                "os_charge"  ,
#                "deep_tau_wp_mtt",
#               ],
#     label=r"$\mu\tau$ signal region no mt, mtt wp, no b veto",
#     aux={'control_reg': "mutau_contol_reg_no_mt_bveto_wp_mtt"}
# )

# mutau_contol_reg_no_mt_bveto_wp_mtt = add_category(
#     config,
#     name="mutau_contol_reg_no_mt_bveto_wp_mtt",
#     id=750 + mutau.id,
#     selection=["cat_mutau"  ,
#                "ss_charge"  ,
#                "deep_tau_wp_mtt",
#               ],
#     label=r"$\mu\tau$ control region no mt, mtt wp, no b veto",
# )


# mutau_signal_reg_no_mt_bveto_wp_mtt = add_category(
#     config,
#     name="mutau_signal_reg_no_mt_bveto",
#     id=800 + mutau.id,
#     selection=["cat_mutau"  ,
#                "os_charge"  ,
#                "deep_tau_wp",
#               ],
#     label=r"$\mu\tau$ signal region no mt, no b veto",
#     aux={'control_reg': "mutau_contol_reg_no_mt_bveto"}
# )

# mutau_contol_reg_no_mt_bveto = add_category(
#     config,
#     name="mutau_contol_reg_no_mt_bveto",
#     id=850 + mutau.id,
#     selection=["cat_mutau"  ,
#                "ss_charge"  ,
#                "deep_tau_wp",
#               ],
#     label=r"$\mu\tau$ control region no mt, no b veto",
# )