import streamlit as st
import pickle
import sys
import os

# --------------------------------
# Project Path
# --------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)


# --------------------------------
# Imports
# --------------------------------

from utils.features import featurize, get_descriptors
from utils.pubchem import get_smiles
from rdkit import Chem


# --------------------------------
# Page Configuration
# --------------------------------

st.set_page_config(
    page_title="MitoTox AI",
    page_icon="🧬",
    layout="wide"
)


# --------------------------------
# Load Model
# --------------------------------

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "mitotox_xgboost_model_final.pkl"
)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


# --------------------------------
# Header
# --------------------------------

st.title("🧬 MitoTox AI")

st.subheader(
    "AI-Powered Mitochondrial Toxicity Screening"
)

st.write(
    """
    Screen small-molecule chemical compounds for potential
    **mitochondrial toxicity** during early-stage drug discovery.
    """
)

st.divider()


# --------------------------------
# Compound Input
# --------------------------------

st.subheader("🔬 Compound Screening")

st.write(
    "Provide a chemical compound using its molecular structure "
    "or compound name."
)

input_type = st.radio(
    "Choose compound input:",
    ["SMILES", "Compound Name"],
    horizontal=True
)


# --------------------------------
# SMILES Input
# --------------------------------

if input_type == "SMILES":

    smiles = st.text_input(
        "Enter SMILES",
        placeholder="Example: CC(=O)OC1=CC=CC=C1C(=O)O"
    )

    st.caption(
        "SMILES allows screening of candidate compounds directly "
        "from their molecular structure."
    )


# --------------------------------
# Compound Name Input
# --------------------------------

else:

    compound_name = st.text_input(
        "Enter Compound Name",
        placeholder="Example: Aspirin"
    )

    st.caption(
        "The compound structure will be retrieved automatically "
        "from PubChem."
    )


st.write("")


# --------------------------------
# Prediction Button
# --------------------------------

predict_button = st.button(
    "🧪 Screen Compound",
    use_container_width=True
)


# --------------------------------
# Screening
# --------------------------------

if predict_button:

    # =================================
    # SMILES SCREENING
    # =================================

    if input_type == "SMILES":

        if smiles.strip() == "":

            st.warning(
                "Please enter a SMILES string."
            )

        else:

            # Validate molecular structure
            mol = Chem.MolFromSmiles(smiles)

            if mol is None:

                st.error(
                    "❌ Invalid SMILES structure."
                )

            else:

                # -----------------------------
                # Feature Extraction
                # -----------------------------

                features = featurize(smiles)

                if features is None:

                    st.error(
                        "❌ Unable to generate molecular features."
                    )

                else:

                    X = features.reshape(1, -1)

                    # -----------------------------
                    # Model Prediction
                    # -----------------------------

                    probability = model.predict_proba(X)[0][1]

                    THRESHOLD = 0.70

                    prediction = 1 if probability >= THRESHOLD else 0

                    # -----------------------------
                    # Molecular Descriptors
                    # -----------------------------

                    descriptors = get_descriptors(smiles)

                    # -----------------------------
                    # Results
                    # -----------------------------

                    st.divider()

                    left, right = st.columns(
                        [1, 1],
                        gap="large"
                    )

                    # =================================
                    # LEFT — PREDICTION
                    # =================================

                    with left:

                        st.subheader(
                            "🩺 Toxicity Screening"
                        )

                        if prediction == 1:

                            st.error(
                                "⚠️ Higher Predicted "
                                "Mitochondrial Toxicity"
                            )

                        else:

                            st.success(
                                "✅ Lower Predicted "
                                "Mitochondrial Toxicity"
                            )

                        st.metric(
                            "Probability of Toxicity",
                            f"{probability:.2%}"
                        )

                        st.progress(
                            float(probability)
                        )

                        st.caption(
                            "Prediction generated by the "
                            "XGBoost mitochondrial toxicity model."
                        )


                    # =================================
                    # RIGHT — DESCRIPTORS
                    # =================================

                    with right:

                        st.subheader(
                            "🧪 Molecular Properties"
                        )

                        col1, col2, col3 = st.columns(3)

                        with col1:

                            st.metric(
                                "Molecular Weight",
                                f"{descriptors['Molecular Weight']:.2f}"
                            )

                        with col2:

                            st.metric(
                                "LogP",
                                f"{descriptors['LogP']:.2f}"
                            )

                        with col3:

                            st.metric(
                                "TPSA",
                                f"{descriptors['TPSA']:.2f}"
                            )


                        col4, col5, col6 = st.columns(3)

                        with col4:

                            st.metric(
                                "H-Bond Donors",
                                descriptors["H-Bond Donors"]
                            )

                        with col5:

                            st.metric(
                                "H-Bond Acceptors",
                                descriptors["H-Bond Acceptors"]
                            )

                        with col6:

                            st.metric(
                                "Rotatable Bonds",
                                descriptors["Rotatable Bonds"]
                            )


    # =================================
    # COMPOUND NAME SCREENING
    # =================================

    else:

        if compound_name.strip() == "":

            st.warning(
                "Please enter a compound name."
            )

        else:

            # -----------------------------
            # Retrieve Structure
            # -----------------------------

            with st.spinner(
                "🔎 Retrieving compound structure..."
            ):

                smiles = get_smiles(compound_name)


            if smiles is None:

                st.error(
                    f"❌ Could not retrieve a compatible "
                    f"molecular structure for '{compound_name}'."
                )

                st.info(
                    """
                    MitoTox AI is designed for small-molecule
                    chemical compounds. Try entering a compound
                    name available in PubChem or use its SMILES
                    representation directly.
                    """
                )

            else:

                # -----------------------------
                # Validate Retrieved Structure
                # -----------------------------

                mol = Chem.MolFromSmiles(smiles)

                if mol is None:

                    st.error(
                        "❌ The retrieved structure could not "
                        "be processed."
                    )

                else:

                    st.success(
                        f"✅ Compound structure retrieved for "
                        f"**{compound_name}**"
                    )

                    # FIXED: Display SMILES as plain code
                    st.write("SMILES:")
                    st.code(smiles, language=None)


                    # -----------------------------
                    # Feature Extraction
                    # -----------------------------

                    features = featurize(smiles)

                    if features is None:

                        st.error(
                            "❌ Unable to generate molecular features."
                        )

                    else:

                        X = features.reshape(1, -1)

                        # -----------------------------
                        # Model Prediction
                        # -----------------------------

                        prediction = model.predict(X)[0]

                        probability = model.predict_proba(X)[0][1]

                        # -----------------------------
                        # Molecular Descriptors
                        # -----------------------------

                        descriptors = get_descriptors(smiles)

                        st.divider()


                        # -----------------------------
                        # Results
                        # -----------------------------

                        left, right = st.columns(
                            [1, 1],
                            gap="large"
                        )


                        # =================================
                        # LEFT — PREDICTION
                        # =================================

                        with left:

                            st.subheader(
                                "🩺 Toxicity Screening"
                            )

                            if prediction == 1:

                                st.error(
                                    "⚠️ Higher Predicted "
                                    "Mitochondrial Toxicity"
                                )

                            else:

                                st.success(
                                    "✅ Lower Predicted "
                                    "Mitochondrial Toxicity"
                                )

                            st.metric(
                                "Probability of Toxicity",
                                f"{probability:.2%}"
                            )

                            st.progress(
                                float(probability)
                            )

                            st.caption(
                                "Prediction generated by the "
                                "XGBoost mitochondrial toxicity model."
                            )


                        # =================================
                        # RIGHT — DESCRIPTORS
                        # =================================

                        with right:

                            st.subheader(
                                "🧪 Molecular Properties"
                            )

                            col1, col2, col3 = st.columns(3)

                            with col1:

                                st.metric(
                                    "Molecular Weight",
                                    f"{descriptors['Molecular Weight']:.2f}"
                                )

                            with col2:

                                st.metric(
                                    "LogP",
                                    f"{descriptors['LogP']:.2f}"
                                )

                            with col3:

                                st.metric(
                                    "TPSA",
                                    f"{descriptors['TPSA']:.2f}"
                                )


                            col4, col5, col6 = st.columns(3)

                            with col4:

                                st.metric(
                                    "H-Bond Donors",
                                    descriptors["H-Bond Donors"]
                                )

                            with col5:

                                st.metric(
                                    "H-Bond Acceptors",
                                    descriptors["H-Bond Acceptors"]
                                )

                            with col6:

                                st.metric(
                                    "Rotatable Bonds",
                                    descriptors["Rotatable Bonds"]
                                )


# --------------------------------
# Footer
# --------------------------------

st.divider()

st.caption(
    "MitoTox AI • AI + Bioinformatics • Early Drug Discovery"
)
