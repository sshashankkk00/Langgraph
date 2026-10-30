# LangGraph Workflows

Examples and experiments for building LLM applications and stateful workflows
with LangGraph and LangChain.

## Contents

- `0_testPython.ipynb` - Python and environment checks
- `conditional.ipynb` - Conditional graph routing
- `iterative.ipynb` - Iterative workflow execution
- `batsman_workflow.ipynb` - A structured workflow example
- `Production.ipynb` - Production-oriented workflow patterns
- `sentiment_response_analysis.ipynb` - Sentiment analysis and response generation

## Requirements

- Python 3.8 or newer
- Jupyter Notebook or JupyterLab
- API keys for any hosted LLM or inference provider used by a notebook

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv myenv
.\myenv\Scripts\Activate.ps1
pip install -r requirement.txt
```

Create a `.env` file in the project root for API keys required by your chosen
notebook. Never commit this file or share its values.

Open the notebooks with:

```powershell
jupyter notebook
```

Run the cells in a notebook from top to bottom. Some notebooks use hosted
inference providers, which may require an active account, valid API token, and
available credits. A `402 Payment Required` response means the provider
account has no remaining inference credits; it is not caused by the notebook
`print` statement.

## Notes

- The local virtual environment (`myenv/`) is intentionally excluded from Git.
- Secrets in `.env` are intentionally excluded from Git.
- Notebook outputs may contain model responses and can be cleared before sharing
  if they include sensitive data.

## License

No license has been specified yet.
