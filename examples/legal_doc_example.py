"""
Example: Legal Document Analysis with DSLA

This example demonstrates how to use the legal_doc adapter
to analyze a contract.
"""

from dsla.adapters.legal_doc import LegalDocAdapter


def main():
    """Run legal document analysis example."""
    print("=" * 80)
    print("DSLA Example: Legal Document Analysis")
    print("=" * 80)
    print()
    
    # Initialize the legal document adapter
    adapter = LegalDocAdapter()
    
    # Sample contract text
    contract_text = """
    EMPLOYMENT AGREEMENT
    
    This Employment Agreement is entered into on January 1, 2024, between
    TechCorp Inc. ("Company") and John Doe ("Employee").
    
    1. TERM: The employment shall commence on January 15, 2024, and continue
    for a period of 2 years, unless terminated earlier as provided herein.
    
    2. COMPENSATION: The Employee shall receive an annual salary of $120,000,
    payable in bi-weekly installments.
    
    3. TERMINATION: Either party may terminate this agreement with 30 days
    written notice. The Company may terminate immediately for cause.
    
    4. CONFIDENTIALITY: Employee agrees to maintain confidentiality of all
    proprietary information and trade secrets during and after employment.
    
    5. NON-COMPETE: Employee agrees not to work for direct competitors for
    12 months following termination of employment within a 50-mile radius.
    """
    
    # Prepare input data
    input_data = {
        "document_type": "employment_agreement",
        "content": contract_text,
        "analysis_focus": "risk assessment and compliance"
    }
    
    # Validate input
    print("1. Validating input...")
    try:
        adapter.validate_input(input_data)
        print("   ✓ Input validation passed")
    except ValueError as e:
        print(f"   ✗ Input validation failed: {e}")
        return
    
    # Adapt input
    print("\n2. Adapting input...")
    adapted_input = adapter.adapt_input(input_data)
    print(f"   Document type: {adapted_input['document_type']}")
    print(f"   Analysis focus: {adapted_input['analysis_focus']}")
    
    # Generate prompt
    print("\n3. Generating prompt...")
    prompt = adapter.format_prompt(**adapted_input)
    print(f"   Prompt length: {len(prompt)} characters")
    print("\n   Prompt preview:")
    print("   " + "-" * 76)
    for line in prompt.split('\n')[:10]:
        print(f"   {line}")
    print("   ...")
    
    # Get available tools
    print("\n4. Available tools:")
    tools = adapter.get_tools()
    for tool in tools:
        print(f"   - {tool.name}: {tool.description}")
    
    # Get schema
    print("\n5. Schema information:")
    schema = adapter.get_schema()
    print(f"   Required input fields: {schema.input_schema.get('required', [])}")
    print(f"   Required output fields: {schema.output_schema.get('required', [])}")
    
    # Simulate output (in real scenario, this would come from LLM)
    print("\n6. Simulating analysis output...")
    simulated_output = {
        "key_clauses": [
            "Employment term: 2 years starting January 15, 2024",
            "Compensation: $120,000 annually",
            "Termination: 30 days notice required, immediate for cause",
            "Confidentiality obligations during and after employment",
            "Non-compete: 12 months, 50-mile radius"
        ],
        "risks": [
            "Non-compete clause may be unenforceable in some jurisdictions",
            "Broad confidentiality terms without clear definition of proprietary information",
            "Immediate termination for cause not clearly defined"
        ],
        "compliance_notes": [
            "Non-compete restrictions should be reviewed against state law",
            "Ensure confidentiality terms comply with trade secret protection laws"
        ],
        "recommendations": [
            "Define 'cause' for immediate termination more specifically",
            "Clarify what constitutes 'proprietary information'",
            "Review non-compete enforceability in relevant jurisdiction",
            "Consider adding arbitration clause for dispute resolution"
        ],
        "summary": "Employment agreement with standard terms. Key areas of concern include non-compete enforceability and need for clearer termination definitions.",
        "document_type": "employment_agreement"
    }
    
    # Adapt output
    adapted_output = adapter.adapt_output(simulated_output)
    
    print("\n7. Analysis Results:")
    print("   " + "=" * 76)
    print(f"\n   Summary: {adapted_output['summary']}")
    
    print(f"\n   Key Clauses ({len(adapted_output['key_clauses'])}):")
    for clause in adapted_output['key_clauses']:
        print(f"   - {clause}")
    
    print(f"\n   Identified Risks ({len(adapted_output['risks'])}):")
    for risk in adapted_output['risks']:
        print(f"   - {risk}")
    
    print(f"\n   Compliance Notes ({len(adapted_output['compliance_notes'])}):")
    for note in adapted_output['compliance_notes']:
        print(f"   - {note}")
    
    print(f"\n   Recommendations ({len(adapted_output['recommendations'])}):")
    for rec in adapted_output['recommendations']:
        print(f"   - {rec}")
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
