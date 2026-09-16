from retrieval_pipeline.api.security import get_password_hash

# Mock user database
# Keys are usernames
MOCK_USERS = {
    # Executive / Corporate
    "ceo_user": {"username": "ceo_user", "hashed_password": get_password_hash("pass123"), "role": "CEO", "department": "Executive_Corporate"},
    "coo_user": {"username": "coo_user", "hashed_password": get_password_hash("pass123"), "role": "COO", "department": "Executive_Corporate"},
    "rmd_user": {"username": "rmd_user", "hashed_password": get_password_hash("pass123"), "role": "Regional_Managing_Directors", "department": "Executive_Corporate"},
    "corp_strat": {"username": "corp_strat", "hashed_password": get_password_hash("pass123"), "role": "Corporate_Strategy", "department": "Executive_Corporate"},
    "corp_comm": {"username": "corp_comm", "hashed_password": get_password_hash("pass123"), "role": "Corporate_Communications", "department": "Executive_Corporate"},
    "pmo_user": {"username": "pmo_user", "hashed_password": get_password_hash("pass123"), "role": "Enterprise_PMO", "department": "Executive_Corporate"},
    
    # Technology & Digital
    "cto_user": {"username": "cto_user", "hashed_password": get_password_hash("pass123"), "role": "CTO", "department": "Technology_Digital"},
    "cio_user": {"username": "cio_user", "hashed_password": get_password_hash("pass123"), "role": "CIO", "department": "Technology_Digital"},
    "ciso_user": {"username": "ciso_user", "hashed_password": get_password_hash("pass123"), "role": "CISO", "department": "Technology_Digital"},
    "cdo_user": {"username": "cdo_user", "hashed_password": get_password_hash("pass123"), "role": "CDO", "department": "Technology_Digital"},
    "cengo_user": {"username": "cengo_user", "hashed_password": get_password_hash("pass123"), "role": "CEngO", "department": "Technology_Digital"},

    # Finance
    "cfo_user": {"username": "cfo_user", "hashed_password": get_password_hash("pass123"), "role": "CFO", "department": "Finance"},
    
    # Sales & Commercial
    "cco_user": {"username": "cco_user", "hashed_password": get_password_hash("pass123"), "role": "CCO", "department": "Sales_Commercial"},
    "cmo_user": {"username": "cmo_user", "hashed_password": get_password_hash("pass123"), "role": "CMO", "department": "Sales_Commercial"},
    
    # Legal & Compliance
    "clo_user": {"username": "clo_user", "hashed_password": get_password_hash("pass123"), "role": "CLO", "department": "Legal_Compliance"},
    
    # People & Culture
    "chro_user": {"username": "chro_user", "hashed_password": get_password_hash("pass123"), "role": "CHRO", "department": "People_Culture"},
}

def get_user(username: str):
    if username in MOCK_USERS:
        return MOCK_USERS[username]
    return None
