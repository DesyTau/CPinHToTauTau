"""
Definition of categories.
"""
import order as od
import law
from columnflow.config_util import add_category
from columnflow.util import maybe_import, DotDict
np = maybe_import("numpy")
def add_categories(config: od.Config,
                   channel = None) -> None:
    
    
    def add_base_categories(config, channel, category_map, base_selection=[]):
        base_cats = []
        base_cat = config.get_category('_'.join(('cat',channel)))      
        for i, (cat_name, cat) in enumerate(category_map.items()):
            kwargs = {
                'name'      : '_'.join((base_cat.name, cat_name)),
                'selection' : base_selection + cat.selection,
                'id'        : 100*(i+1)+ base_cat.id,
                'label'     :' '.join((base_cat.label.split(' ')[0], cat.label if 'label' in cat else cat_name))
            }
            if 'aux' in cat.keys():
                kwargs['aux'] = {}
                for (aux_spec, aux_content) in cat.aux.items():
                    if 'regs' in aux_spec:
                        kwargs['aux'][aux_spec] = {key: '_'.join((base_cat.name, val)) for (key,val) in aux_content.items()}
                    else:
                        if aux_spec == 'fit_var' and isinstance(aux_content, str):
                            aux_content = [aux_content]
                        kwargs['aux'][aux_spec] = aux_content
            base_cats.append(kwargs['name'])
            add_category(config, **kwargs)
        return base_cats
            
    
    def add_child_category(config, parent_cat, child_cat, child_name):
        max_cat_id = np.max(config.categories.ids())
        kwargs = {
                    'name'      : '__'.join((parent_cat.name, child_name)),
                    'selection' : parent_cat.selection + child_cat.selection,
                    'id'        : int(max_cat_id+1),
                    'label'     : ' '.join((parent_cat.label, child_cat.label if 'label' in child_cat else child_name))
                }
        if parent_cat.aux:
            kwargs['aux'] = {}
            for (aux_key, aux_content) in parent_cat.aux.items():
                if 'regs' in aux_key:
                    reg_map = parent_cat.aux[aux_key]
                    add_tag = lambda cat_name, tag=child_name : '__'.join((cat_name, tag))
                    reg_map_tagged = dict(zip(reg_map.keys(), map(add_tag, reg_map.values())))
                    kwargs['aux'][aux_key] = reg_map_tagged
                else:
                    kwargs['aux'][aux_key] = aux_content
        if 'aux' in child_cat.keys():
            if 'aux' not in kwargs:
                kwargs['aux'] = {}
            for (aux_key, aux_content) in child_cat['aux'].items():
                if aux_key == 'fit_var' and isinstance(aux_content, str):
                    aux_content = [aux_content]
                kwargs['aux'][aux_key] = aux_content
                
        add_category(config, **kwargs)
        return kwargs['name']
    
    
    def create_child_categories(config, parent_categories, child_category_map):
        out_cats = []
        for cat_name in parent_categories:
            #skip 0-level categories that are used to define channelss
            if cat_name in ['incl', 'cat_mutau', 'cat_etau']: continue
            parent_cat = config.get_category(cat_name)
            for child_name, child_cat in child_category_map.items():
                full_name = add_child_category(config, parent_cat, child_cat, child_name)
                out_cats.append(full_name)
        return out_cats
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
    if channel=='mutau':
        add_category(
            config,
            name="cat_mutau",
            id=2,
            selection=["cat_mutau"],
            label=r"$\mu\tau$ inclusive",)
        
        
    if channel=='tautau':
        add_category(
            config,
            name="cat_tautau",
            id=2,
            selection=["cat_tautau"],
            label=r"$\mu\tau$ inclusive",)  
    if channel=='etau':
        add_category(
            config,
            name="cat_etau",
            id=3,
            selection=["cat_etau"],
            label=r"$e\tau$ inclusive")
    if channel=='emu':
        add_category(
            config,
            name="cat_emu",
            id=4,
            selection=["cat_emu"],
            label=r"$e\mu$ inclusive")
    
    #Define initial category map with selections and call the function
    #Don't change this part: it is important for fake factor method    
    base_selection = [f'cat_{channel}']
    
    category_map  = DotDict.wrap({
        "sr"            : { 'selection' : ["lep_iso", "os_charge", "D_zeta_cut"],
                            'label'     : "signal region",
                            'aux'       : {
                                           #qcd estimation categories
                                           'abcd_regs' : {
                                               'ar'    : 'abcd_ar',
                                               'dr_num': 'abcd_dr_num',
                                               'dr_den': 'abcd_dr_den',
                                               },
                                        #    #fake factor categories
                                        #    'ff_regs': {
                                        #        "ar_qcd"      : "ar_qcd",
                                        #        "dr_num_qcd"  : "dr_num_qcd",
                                        #        "dr_den_qcd"  : "dr_den_qcd",
                                        #        "ar_yields"   : "ar_yields",
                                        #        #categories for closure tests
                                        #        "dr_den_qcd_w_ff": "dr_den_qcd_w_ff",
                                        #    },
                                           },},
        #categories for jet fakes estimation via classic Fake Factor method    
        # "ar_qcd"         : {'selection' : ["lep_iso"    , "ss_charge"],
        #                     'aux'       : {'apply_ff': ''}}, #qcd
        # "dr_num_qcd"     : {'selection' : ["lep_inv_iso", "os_charge"],},
        # "dr_den_qcd"     : {'selection' : ["lep_inv_iso", "ss_charge"],},
        # "dr_den_qcd_w_ff": {'selection' : ["lep_inv_iso", "os_charge"],
        #                     'aux'       : {'apply_ff': ''}},
        # "ar_yields"      : {'selection' : ["lep_iso", "os_charge"],},
        #categories for QCD estimation via classic ABCD method 
        "abcd_ar"       : { 'selection' : ["lep_iso", "ss_charge"], 'label' : "same sign region"},
        "abcd_dr_num"   : { 'selection' : ["lep_inv_iso", "os_charge"]},
        "abcd_dr_den"   : { 'selection' : ["lep_inv_iso", "ss_charge"]},
        
        # "sr_no_mt"      : { 'selection' : ["lep_iso", "os_charge"],
        #                     'label'     : "signal region no mt",
        #                     'aux'       : {
        #                                    #qcd estimation categories
        #                                    'abcd_regs' : {
        #                                        'ar'    :  'abcd_ar_no_mt',
        #                                        'dr_num':  'abcd_dr_num_no_mt',
        #                                        'dr_den':  'abcd_dr_den_no_mt',
        #                                        },
        #                                    },},
        # #categories for QCD estimation via classic ABCD method 
        # "abcd_ar_no_mt"       : { 'selection' : ["lep_iso", "ss_charge"], 'label' : "ss region no mt"},
        # "abcd_dr_num_no_mt"   : { 'selection' : ["lep_inv_iso", "os_charge"]},
        # "abcd_dr_den_no_mt"   : { 'selection' : ["lep_inv_iso", "ss_charge"]},
    })
    
    add_base_categories(config, channel, category_map, base_selection)
    
    from MSSM_H_tt.config.mass_points import read_bdt_masses

    MASS_POINTS = read_bdt_masses()

    # Merged three-region BDT approach.
    #
    # The selections below are produced dynamically in categorization/main.py.
    # They use the raw four-class probabilities from the BDT-score producer:
    #
    #   P_sig = P_ggphi + P_bbphi
    #
    # and define the fit regions through:
    #
    #   signal region : P_sig is larger than both P_DY and P_TT
    #   DY region     : P_DY  is larger than both P_sig and P_TT
    #   TT region     : P_TT  is larger than both P_sig and P_DY
    #
    # This matches the 10-feature training setup, whose adaptive post-processing
    # produces the signal-region flattened 2D variables:
    #
    #   bdt_D_sig_vs_Disc_ggphi_M{mass}
    #   bdt_D_sig_vs_Disc_bbphi_M{mass}
    #
    # The old standalone ggphi and bbphi four-class fit categories are not added
    # here, because they would overlap with the merged signal-like region. They
    # can still exist as diagnostic categorizers in categorization/main.py.

    bdt_cats_map = DotDict.wrap({})

    bdt_regions = {
        "ggphi_and_bbphi": {
            "label": "ggϕ + bbϕ",
            "selection": "bdt_cat_ggphi_and_bbphi",
            "fit_var": [
                "D_sig_vs_Disc_ggphi",
                "D_sig_vs_Disc_bbphi",
            ],
        },
        "dy": {
            "label": "DY",
            "selection": "bdt_cat_dy",
            "fit_var": "D_DY",
        },
        "tt": {
            "label": "tt̄",
            "selection": "bdt_cat_tt",
            "fit_var": "D_TT",
        },
    }

    for m in MASS_POINTS:
        for region, spec in bdt_regions.items():
            fit_vars = spec["fit_var"]

            if isinstance(fit_vars, str):
                fit_vars = [fit_vars]

            fit_vars = [
                f"bdt_{fit_var}_M{m}"
                for fit_var in fit_vars
            ]

            bdt_cats_map[f"bdt_{region}_M{m}"] = DotDict.wrap({
                "selection": [
                    f"{spec['selection']}_M{m}",
                ],
                "label": f"BDT cat. {spec['label']} (M={m})",
                "aux": {
                    "fit_var": fit_vars,
                },
            })

    bdt_parent_categories = [
        f"cat_{channel}_sr",
        f"cat_{channel}_abcd_ar",
        f"cat_{channel}_abcd_dr_num",
        f"cat_{channel}_abcd_dr_den",
        ]
    
    create_child_categories(
        config,
        parent_categories=bdt_parent_categories,
        child_category_map=bdt_cats_map,
    )

    # create_child_categories(
    #     config,
    #     parent_categories=config.categories.names(),
    #     child_category_map=bdt_cats_map,
    # )
