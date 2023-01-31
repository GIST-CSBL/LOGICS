
import pandas as pd
import numpy as np
from io import StringIO
import urllib
import json
import re
from . import chemistry

def get_SMILES_PubChemCID(cidlist, request_size=50):
    """
        This function downloads the SMILES by the cid list.
        The http request size (batch size) should be small; recommended to be < 100.
    """
    cidlist = [str(cid) for cid in cidlist]
    num_requests = int(len(cidlist)/request_size)+1
    batches = np.array_split(cidlist, num_requests)
    smiles_list = []
    for batch in batches:
        url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"+",".join(batch)+"/property/CanonicalSMILES/txt"
        opened = urllib.request.urlopen(url)
        read_decoded = opened.read().decode("utf-8")
        batch_smiles = read_decoded.split('\n')[:-1]
        # somehow, the decoded string has a single \n at the end, causing the last element to be empty ''. Thus, [:-1]
        smiles_list.extend(batch_smiles)
    return smiles_list

class PubChemAssaysEntrezGene:
    """
    This class was originally designed to process PubChem assay APIs, but you can use the
    utility functions of this class by manually assign DFs for raw and filtered after defining an emtpy object.

    Every row-deletion operation performs reset_index() at the end.

    >> raw table columns of PubChem response:
    ['acname', 'acqualifier', 'activityid', 'acvalue', 'aid', 'aidmdate',
       'aidname', 'aidsrcname', 'aidtypeid', 'baid', 'cellids', 'cid',
       'cmpdname', 'geneid', 'hasdrc', 'pmid', 'protacxn', 'repacxn', 'rnai',
       'sid', 'targetname', 'targettaxid', 'targeturl', 'taxids']

    download file max size limited to: 1000000
    """
    default_interest = ['acname', 'acqualifier', 'acvalue', 'aid', 'aidname', 'cid']

    def __init__(self, entrezid=None):
        self.entrezid = entrezid
        if entrezid == None:
            self.url, self.raw, self.filtered = None, None, None
            return
        self.url = "https://pubchem.ncbi.nlm.nih.gov/sdq/sdqagent.cgi?infmt=json&outfmt=jsonp&query={%22download%22:%22*%22,%22collection%22:%22bioactivity%22,%22where%22:{%22ands%22:[{%22geneid%22:%22"+str(entrezid)+"%22},{%22acvalue%22:%22notnull%22},{%22cid%22:%22notnull%22}]},%22order%22:[%22relevancescore,desc%22],%22start%22:1,%22limit%22:1000000}"
        opened = urllib.request.urlopen(self.url)
        read_decoded = opened.read().decode("utf-8")
        try:
            self.raw = pd.read_json(StringIO(read_decoded))
        except ValueError as verr:
            print(verr)
            print("--- This error is usually caused by the wrong JSON format response from NCBI API.")
            print("--- The text response is saved to: pubchem_%d.txt"%entrezid)
            print("--- Fix the problem of the file, and call load_json_response() to manually initialize.")
            print("--- https://jsonlint.com/")
            with open('pubchem_%d.txt'%entrezid, 'w', encoding='utf-8') as f:
                f.write(read_decoded)
            self.raw, self.filtered = None, None
            return
        self.filtered = self.raw.copy()

    def load_json_response(self, fpath):
        with open(fpath, 'r', encoding='utf-8') as f:
            assay_dict = json.load(f)
        self.raw = pd.DataFrame(assay_dict)
        self.raw['acvalue'] = self.raw['acvalue'].astype(float)
        self.filtered = self.raw.copy()
    
    def get_raw(self): return self.raw
    def get_filtered(self): return self.filtered

    def reset_filter(self):
        self.filtered = self.raw.copy()
    
    def filter_set_default_columns(self):
        self.filtered = self.filtered[self.default_interest]

    def filter_set_exactly(self, col, exact_val):
        self.filtered = self.filtered[self.filtered[col]==exact_val].reset_index(drop=True)

    def variant_loc(self, df: pd.DataFrame):
        """
            return location (indices) where the assay was done on the protein variants,
            based on 'aidname'.
        """
        aidnames = df['aidname']
        pattern = "[^A-Za-z0-9]+[A-Z]{1}[0-9]+[A-Z]{1}[^A-Za-z0-9]+"
        compiled = re.compile(pattern)
        variant_loc = []
        for i in range(len(aidnames)):
            if len(compiled.findall(aidnames[i])) > 0:
                variant_loc.append(i)
        return variant_loc

    def filter_del_variant(self):
        variant_loc = self.variant_loc(self.filtered)
        self.filtered = self.filtered.drop(variant_loc).reset_index(drop=True)

    def filter_append_smiles_download(self):
        """ this also performs the canonicalization of the SMILES """
        smiles_list = get_SMILES_PubChemCID(self.filtered['cid'].tolist())
        vcsmiles, invalid_ids = chemistry.get_valid_canons(smiles_list)
        if len(invalid_ids) > 0:
            print("invalid SMILES is detected in the PubChem data!")
            print("pos: ", invalid_ids)
            print("invalid SMILES list:")
            for smi in smiles_list:
                print(smi)
            return smiles_list, invalid_ids
        self.filtered['smiles'] = vcsmiles # append canonical SMILES column

    def filter_del_disconnected_smiles(self):
        """ 
            In SMILES, disconnected components are specified by period '.' 
            e.g. [Na+].[O-]c1ccccc1 or c1cc([O-].[Na+])ccc1
            Remove these kinds of SMILES data.
        """
        self.filtered = self.filtered[~self.filtered['smiles'].str.contains('\.')].reset_index(drop=True)

    def filter_append_median(self, median_of='acvalue', groupby='smiles'):
        grouped_median = self.filtered.groupby(groupby).median() # index: smiles, columns: all numeric columns having median values
        grouped_median = grouped_median[[median_of]].rename(columns={median_of:'med_'+median_of}) # new column is m_acvalue
        self.filtered = self.filtered.merge(grouped_median, on=groupby)

        
