import requests


def get_smiles(compound_name):
    """
    Retrieve a compound's canonical SMILES from PubChem.

    Parameters:
        compound_name (str): Name of the chemical compound.

    Returns:
        str: Canonical SMILES if found.
        None: If the compound cannot be found.
    """

    if not compound_name or compound_name.strip() == "":
        return None

    compound_name = compound_name.strip()

    # Correct PubChem PUG REST API URL
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
        f"compound/name/{requests.utils.quote(compound_name)}/"
        "property/ConnectivitySMILES/JSON"
    )

    try:
        response = requests.get(
            url,
            timeout=10
        )

        if response.status_code != 200:
            return None

        data = response.json()

        properties = data.get("PropertyTable", {}).get("Properties", [])

        if not properties:
            return None

        smiles = properties[0].get("ConnectivitySMILES")

        return smiles

    except (requests.RequestException, ValueError, KeyError, TypeError):
        return None
    