"""Generate a small sample PDF (a fictional employee handbook) for demos and tests."""
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

PAGES = [
    (
        "Northwind Labs Employee Handbook",
        [
            "Welcome to Northwind Labs. This handbook explains our policies on time off, remote work, "
            "expenses, and equipment. It was last updated in January 2026.",
            "Northwind Labs was founded in 2018 and is headquartered in Pune, India. The company has "
            "about 240 employees across three offices in Pune, Bengaluru, and Dehradun.",
        ],
    ),
    (
        "Time Off and Leave",
        [
            "Full-time employees receive 24 days of paid annual leave per calendar year. Unused leave "
            "up to 10 days can be carried over into the next year.",
            "Sick leave is separate from annual leave. Each employee gets 12 days of paid sick leave per "
            "year. A medical certificate is required for absences longer than 3 consecutive days.",
            "Parental leave is 26 weeks for the primary caregiver and 4 weeks for the secondary caregiver.",
        ],
    ),
    (
        "Remote Work Policy",
        [
            "Employees may work remotely up to 3 days per week with manager approval. Everyone is "
            "expected to be in the office on Tuesdays and Wednesdays for team collaboration.",
            "Fully remote arrangements require approval from the department head and must be reviewed "
            "every 6 months. Remote employees receive a one-time home office stipend of 15,000 rupees.",
        ],
    ),
    (
        "Expenses and Travel",
        [
            "Business expenses must be submitted within 30 days through the Expense Portal. Receipts are "
            "mandatory for any expense above 500 rupees.",
            "Domestic flights must be booked in economy class at least 7 days in advance. Hotel stays are "
            "capped at 6,000 rupees per night in metro cities and 4,000 rupees elsewhere.",
            "The daily meal allowance during business travel is 1,200 rupees.",
        ],
    ),
    (
        "Equipment and Security",
        [
            "New employees receive a company laptop on their first day. Laptops are replaced every 4 "
            "years. Lost or stolen equipment must be reported to the IT Helpdesk within 24 hours.",
            "All employees must enable two-factor authentication and complete the annual security "
            "awareness training by March 31 each year. Passwords must be at least 14 characters long.",
        ],
    ),
]


def main(out: str = "sample_docs/northwind_handbook.pdf") -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    story = []
    for i, (title, paras) in enumerate(PAGES):
        story.append(Paragraph(title, styles["Heading1"]))
        story.append(Spacer(1, 12))
        for p in paras:
            story.append(Paragraph(p, styles["BodyText"]))
            story.append(Spacer(1, 10))
        if i < len(PAGES) - 1:
            story.append(PageBreak())
    SimpleDocTemplate(out, pagesize=letter).build(story)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
