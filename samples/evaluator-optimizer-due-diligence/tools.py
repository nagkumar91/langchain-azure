"""Mock research tools for the Due Diligence Report Generator.

All tools return realistic but synthetic data so the sample runs
without external API dependencies.
"""

from __future__ import annotations

from langchain_core.tools import tool


# ============================================================
# Financial Analysis Tools
# ============================================================


@tool
def get_financial_statements(company_name: str, years: int = 3) -> dict:
    """Retrieve historical financial statements for a company.

    Args:
        company_name: Name of the company to research.
        years: Number of years of history to retrieve.
    """
    return {
        "company": company_name,
        "period": f"Last {years} years",
        "revenue": [
            {"year": 2023, "amount_millions": 4850, "growth_pct": 12.3},
            {"year": 2024, "amount_millions": 5620, "growth_pct": 15.9},
            {"year": 2025, "amount_millions": 6340, "growth_pct": 12.8},
        ][:years],
        "ebitda_margin_pct": [28.5, 31.2, 33.1][:years],
        "net_income_millions": [580, 720, 890][:years],
        "free_cash_flow_millions": [420, 550, 680][:years],
    }


@tool
def get_balance_sheet(company_name: str) -> dict:
    """Retrieve current balance sheet data for a company.

    Args:
        company_name: Name of the company to research.
    """
    return {
        "company": company_name,
        "total_assets_millions": 12400,
        "total_liabilities_millions": 5800,
        "total_equity_millions": 6600,
        "cash_and_equivalents_millions": 1850,
        "total_debt_millions": 3200,
        "debt_to_equity_ratio": 0.48,
        "current_ratio": 2.1,
        "quick_ratio": 1.7,
    }


@tool
def get_valuation_metrics(company_name: str) -> dict:
    """Retrieve valuation metrics and multiples for a company.

    Args:
        company_name: Name of the company to research.
    """
    return {
        "company": company_name,
        "market_cap_millions": 18500,
        "enterprise_value_millions": 19850,
        "pe_ratio": 24.3,
        "forward_pe_ratio": 19.8,
        "ev_ebitda": 14.2,
        "ev_revenue": 3.1,
        "price_to_book": 2.8,
        "peg_ratio": 1.4,
        "dcf_implied_value_per_share": 142.50,
        "current_price_per_share": 128.75,
        "analyst_consensus": "overweight",
        "price_target_median": 155.00,
    }


# ============================================================
# Market Analysis Tools
# ============================================================


@tool
def get_market_size(industry: str) -> dict:
    """Retrieve total addressable market data for an industry.

    Args:
        industry: The industry sector to research.
    """
    return {
        "industry": industry,
        "tam_billions_2025": 185.0,
        "tam_billions_2030_projected": 340.0,
        "cagr_pct": 12.9,
        "sam_billions": 65.0,
        "som_billions": 8.2,
        "key_growth_drivers": [
            "Digital transformation acceleration",
            "Regulatory compliance requirements",
            "AI and automation adoption",
            "Geographic expansion into emerging markets",
        ],
        "market_maturity": "growth_phase",
    }


@tool
def get_competitive_landscape(company_name: str, industry: str) -> dict:
    """Retrieve competitive landscape analysis for a company.

    Args:
        company_name: The target company.
        industry: The industry sector.
    """
    return {
        "company": company_name,
        "industry": industry,
        "market_share_pct": 12.5,
        "market_position": "top_5",
        "competitors": [
            {"name": "AlphaCorp", "market_share_pct": 22.0, "strength": "scale"},
            {"name": "BetaTech", "market_share_pct": 18.5, "strength": "technology"},
            {
                "name": "GammaSolutions",
                "market_share_pct": 15.0,
                "strength": "distribution",
            },
            {"name": company_name, "market_share_pct": 12.5, "strength": "innovation"},
            {"name": "DeltaGroup", "market_share_pct": 8.0, "strength": "cost_leadership"},
        ],
        "barriers_to_entry": ["high", "regulatory_licenses", "network_effects", "switching_costs"],
        "competitive_moat": "moderate",
        "disruption_risk": "medium",
    }


@tool
def get_industry_trends(industry: str) -> dict:
    """Retrieve key industry trends and forecasts.

    Args:
        industry: The industry sector to research.
    """
    return {
        "industry": industry,
        "key_trends": [
            {"trend": "AI/ML Integration", "impact": "high", "timeline": "1-3 years"},
            {"trend": "Sustainability Mandates", "impact": "medium", "timeline": "2-5 years"},
            {"trend": "Consolidation Wave", "impact": "high", "timeline": "1-2 years"},
            {"trend": "Platform Business Models", "impact": "medium", "timeline": "ongoing"},
        ],
        "technology_disruption_score": 7.2,
        "regulatory_change_score": 6.5,
        "consumer_behavior_shift_score": 5.8,
    }


# ============================================================
# Risk Analysis Tools
# ============================================================


@tool
def get_regulatory_risks(company_name: str, industry: str) -> dict:
    """Retrieve regulatory and compliance risk assessment.

    Args:
        company_name: The target company.
        industry: The industry sector.
    """
    return {
        "company": company_name,
        "industry": industry,
        "regulatory_environment": "moderately_complex",
        "key_regulations": [
            {
                "name": "Data Privacy (GDPR/CCPA)",
                "compliance_status": "compliant",
                "risk_level": "medium",
            },
            {
                "name": "Industry-Specific Licensing",
                "compliance_status": "compliant",
                "risk_level": "low",
            },
            {
                "name": "Anti-Trust Scrutiny",
                "compliance_status": "under_review",
                "risk_level": "high",
            },
            {
                "name": "ESG Disclosure Requirements",
                "compliance_status": "partial",
                "risk_level": "medium",
            },
        ],
        "pending_litigation": 2,
        "litigation_exposure_millions": 45,
        "regulatory_change_risk": "medium",
    }


@tool
def get_operational_risks(company_name: str) -> dict:
    """Retrieve operational risk assessment for a company.

    Args:
        company_name: The target company.
    """
    return {
        "company": company_name,
        "key_person_risk": "medium",
        "supply_chain_concentration": {
            "top_supplier_pct": 18,
            "single_source_components": 2,
            "geographic_concentration": "moderate",
        },
        "technology_risk": {
            "legacy_systems_pct": 15,
            "cybersecurity_maturity": "above_average",
            "tech_debt_score": 3.2,
        },
        "workforce_risks": {
            "turnover_rate_pct": 12.5,
            "key_talent_retention": "good",
            "succession_planning": "adequate",
        },
        "business_continuity_score": 7.8,
    }


@tool
def get_financial_risks(company_name: str) -> dict:
    """Retrieve financial risk indicators for a company.

    Args:
        company_name: The target company.
    """
    return {
        "company": company_name,
        "currency_exposure": {
            "revenue_foreign_pct": 35,
            "hedging_strategy": "partial",
            "key_currencies": ["EUR", "GBP", "JPY"],
        },
        "interest_rate_sensitivity": "moderate",
        "credit_risk": {
            "credit_rating": "BBB+",
            "outlook": "stable",
            "default_probability_5yr_pct": 0.8,
        },
        "liquidity_risk": "low",
        "customer_concentration": {
            "top_10_customers_revenue_pct": 42,
            "largest_customer_pct": 8.5,
        },
    }


# ============================================================
# ESG Analysis Tools
# ============================================================


@tool
def get_environmental_data(company_name: str) -> dict:
    """Retrieve environmental metrics and sustainability data.

    Args:
        company_name: The target company.
    """
    return {
        "company": company_name,
        "carbon_footprint": {
            "scope_1_tons_co2": 12500,
            "scope_2_tons_co2": 28000,
            "scope_3_tons_co2": 85000,
            "reduction_target_pct": 30,
            "target_year": 2030,
        },
        "energy_usage": {
            "renewable_pct": 45,
            "total_mwh": 125000,
            "efficiency_trend": "improving",
        },
        "waste_management": {
            "recycling_rate_pct": 72,
            "zero_waste_facilities_pct": 25,
        },
        "water_usage_cubic_meters": 450000,
        "environmental_incidents_last_3yr": 1,
    }


@tool
def get_social_metrics(company_name: str) -> dict:
    """Retrieve social responsibility and workforce metrics.

    Args:
        company_name: The target company.
    """
    return {
        "company": company_name,
        "diversity_inclusion": {
            "women_in_leadership_pct": 35,
            "ethnic_diversity_index": 0.62,
            "pay_equity_ratio": 0.97,
            "dei_program": True,
        },
        "labor_practices": {
            "living_wage_compliance": True,
            "supply_chain_audits_per_year": 24,
            "worker_safety_incident_rate": 1.2,
            "union_relations": "constructive",
        },
        "community_impact": {
            "charitable_giving_millions": 12.5,
            "volunteer_hours": 45000,
            "local_hiring_pct": 68,
        },
        "employee_satisfaction_score": 4.1,
        "glassdoor_rating": 3.9,
    }


@tool
def get_governance_data(company_name: str) -> dict:
    """Retrieve corporate governance metrics and practices.

    Args:
        company_name: The target company.
    """
    return {
        "company": company_name,
        "board_structure": {
            "board_size": 11,
            "independent_directors_pct": 72,
            "average_tenure_years": 6.3,
            "diversity_pct": 36,
            "separate_chair_ceo": True,
        },
        "executive_compensation": {
            "ceo_total_comp_millions": 14.2,
            "ceo_pay_ratio": 185,
            "performance_linked_pct": 65,
            "clawback_policy": True,
        },
        "transparency": {
            "audit_quality": "big_4",
            "financial_restatements_3yr": 0,
            "whistleblower_program": True,
            "lobbying_disclosure": "partial",
        },
        "shareholder_rights": {
            "dual_class_shares": False,
            "poison_pill": False,
            "proxy_access": True,
        },
        "esg_ratings": {
            "msci": "A",
            "sustainalytics_risk_score": 22.5,
            "cdp_climate_score": "B",
        },
    }


# ============================================================
# Tool collections by analyst type
# ============================================================

FINANCIAL_TOOLS = [get_financial_statements, get_balance_sheet, get_valuation_metrics]
MARKET_TOOLS = [get_market_size, get_competitive_landscape, get_industry_trends]
RISK_TOOLS = [get_regulatory_risks, get_operational_risks, get_financial_risks]
ESG_TOOLS = [get_environmental_data, get_social_metrics, get_governance_data]

ALL_TOOLS = FINANCIAL_TOOLS + MARKET_TOOLS + RISK_TOOLS + ESG_TOOLS

TOOLS_BY_ANALYST = {
    "financial": FINANCIAL_TOOLS,
    "market": MARKET_TOOLS,
    "risk": RISK_TOOLS,
    "esg": ESG_TOOLS,
}
