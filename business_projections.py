import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline

class FrontiersFinance:
    def __init__(self):
        self.years = np.array(range(1, 11))  # Years 1 to 10

        self.plot = False

        self.df_financial = pd.read_csv('models/model-projections-financial.csv')
        self.df_numeric = pd.read_csv('models/model-projections-numeric.csv')

        # Extract number of participating schools
        self.num_schools = self.initialize("Members", "Total", True)

        # Revenue projections
        self.revenue_median = self.initialize("Total Income", "Median")
        self.revenue_high = self.initialize("Total Income", "High")
        self.revenue_low = self.initialize("Total Income", "Low")

        # Expense projections
        self.expense_median = self.initialize("Total Expense", "Median")
        self.expense_high = self.initialize("Total Expense", "High")
        self.expense_low = self.initialize("Total Expense", "Low")

        # Profit projections
        self.profit_median = self.initialize("Total Profit", "Median")
        self.profit_high = self.initialize("Total Profit", "High")
        self.profit_low = self.initialize("Total Profit", "Low")

        # Membership tier revenue
        self.revenue_basic = self.initialize("Memberships", "Basic")
        self.revenue_advanced = self.initialize("Memberships", "Advanced")
        self.revenue_premium = self.initialize("Memberships", "Premium")

        # Institution counts by tier (from numeric model)
        self.members_basic = self.initialize("Members", "Basic", True)
        self.members_advanced = self.initialize("Members", "Advanced", True)
        self.members_premium = self.initialize("Members", "Premium", True)

        # Aggregated revenue categories
        self.revenue_memberships = self.initialize("Memberships", "Total Median")
        self.revenue_sponsorships = self.initialize("Sponsorships", "Total Median")
        self.revenue_grants = self.initialize("Grants", "Total Median")
        self.revenue_licensing = self.initialize("Licensing", "Total Median")
        self.revenue_ventures = self.initialize("Ventures", "Total Median")

        # Expense category breakdowns
        self.expense_salaries = self.initialize("Salaries", "Total Median")
        self.expense_mentorship = self.initialize("Mentorship", "Total Median")
        self.expense_infrastructure = self.initialize("Infrastructure", "Total Median")
        self.expense_marketing = self.initialize("Marketing", "Total Median")
        self.expense_events = self.initialize("Events", "Total Median")

        # Define color dictionary for consistent chart styling
        self.colors = {
            "Total Revenue": "#AFCBFF",  # Soft Blue
            "Total Expense": "#FFADAD",  # Soft Red
            "Profit": "#228B22",  # Forest Green
            "Profit Margin": "#333333",  # Dark Gray

            "Revenue Growth Rate": "#AFCBFF",  # Soft Blue
            "Expense Growth Rate": "#FFADAD",  # Soft Red

            "Sponsorship Revenue": "#A0C4FF",  # Sky Blue
            "Membership Revenue": "#C9E264",  # Olive Pastel

            "Basic Membership": "#D4F8E8",  # Pale Green
            "Advanced Membership": "#A1C181",  # Soft Lime
            "Premium Membership": "#728C69",  # Muted Forest

            "Staff Expense": "#FFB3BA",  # Warm Red
            "Mentorship Expense": "#FFD6A5",  # Peach
            "Infrastructure Expense": "#C5A3FF",  # Lavender
            "Marketing Expense": "#C4A484",  # Warm Brown
            "Event Expense": "#FFB5E8",  # Soft Pink

            "Venture Revenue": "#C2C5FF",  # Muted Indigo
            "Licensing Revenue": "#B5EAD7",  # Teal Pastel
            "Grants Revenue": "#D9E4F5"  # Pale Gray-Blue
        }

    def initialize(self, category, subcategory, numeric=False):
        if numeric:
            return self.df_numeric[(self.df_numeric['Category'] == category) &
                                    (self.df_numeric['Subcategory'] == subcategory)
                                    ].iloc[0, 2:].to_numpy(dtype=int)
        else:
            return self.df_financial[(self.df_financial['Category'] == category) &
                                     (self.df_financial['Subcategory'] == subcategory)
                                     ].iloc[0, 2:].to_numpy(dtype=int)

    def plot_cost_efficiency(self):
        """Green line represents total expenses divided by the number of member institutions."""
        cost_per_member = self.expense_median / self.num_schools

        plt.figure(figsize=(10, 6))

        # Plot cost efficiency per member
        plt.plot(self.years, cost_per_member, marker='o', linestyle='-', color=self.colors["Profit"],
                 label="Cost per Member")

        plt.xlabel("Year")
        plt.ylabel("Per-Member Cost (¥M)")
        plt.title("Cost Efficiency per Member")
        plt.legend()
        plt.grid(True)
        plt.xticks(self.years)

        if self.plot:
            plt.show()
        self.save_plot("cost_efficiency")

    def plot_expense_breakdown(self):
        """Stacked bar chart displaying how expenses are distributed across different categories."""
        plt.figure(figsize=(10, 6))

        # Plot actual categories in stacked order
        bottom = np.zeros_like(self.expense_salaries)

        plt.bar(self.years, self.expense_salaries, bottom=bottom, label="Salaries", color=self.colors["Staff Expense"])
        bottom += self.expense_salaries

        plt.bar(self.years, self.expense_mentorship, bottom=bottom, label="Mentorship",
                color=self.colors["Mentorship Expense"])
        bottom += self.expense_mentorship

        plt.bar(self.years, self.expense_infrastructure, bottom=bottom, label="Infrastructure",
                color=self.colors["Infrastructure Expense"])
        bottom += self.expense_infrastructure

        plt.bar(self.years, self.expense_marketing, bottom=bottom, label="Marketing",
                color=self.colors["Marketing Expense"])
        bottom += self.expense_marketing

        plt.bar(self.years, self.expense_events, bottom=bottom, label="Events", color=self.colors["Event Expense"])

        plt.xlabel("Year")
        plt.ylabel("Expense (¥M)")
        plt.title("Expense Breakdown By Source")
        plt.xticks(self.years)
        plt.legend()
        plt.grid(True, axis='y')

        if self.plot:
            plt.show()
        self.save_plot("expense_breakdown")

    def plot_expense_breakdown_proportional(self):
        """Normalized stacked bar chart showing expense allocation as percentages."""
        total_expense = (
                self.expense_salaries +
                self.expense_mentorship +
                self.expense_infrastructure +
                self.expense_marketing +
                self.expense_events
        )

        plt.figure(figsize=(10, 6))

        bottom = np.zeros_like(self.expense_salaries, dtype=float)

        plt.bar(self.years, (self.expense_salaries / total_expense) * 100, bottom=bottom,
                label="Salaries", color=self.colors["Staff Expense"])
        bottom += (self.expense_salaries / total_expense) * 100

        plt.bar(self.years, (self.expense_mentorship / total_expense) * 100, bottom=bottom,
                label="Mentorship", color=self.colors["Mentorship Expense"])
        bottom += (self.expense_mentorship / total_expense) * 100

        plt.bar(self.years, (self.expense_infrastructure / total_expense) * 100, bottom=bottom,
                label="Infrastructure", color=self.colors["Infrastructure Expense"])
        bottom += (self.expense_infrastructure / total_expense) * 100

        plt.bar(self.years, (self.expense_marketing / total_expense) * 100, bottom=bottom,
                label="Marketing", color=self.colors["Marketing Expense"])
        bottom += (self.expense_marketing / total_expense) * 100

        plt.bar(self.years, (self.expense_events / total_expense) * 100, bottom=bottom,
                label="Events", color=self.colors["Event Expense"])

        plt.xlabel("Year")
        plt.ylabel("Expense Allocation (%)")
        plt.title("Expense Breakdown By Source (Proportional)")
        plt.xticks(self.years)
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(True, axis='y')

        if self.plot:
            plt.show()
        self.save_plot("expense_breakdown_proportional")

    def plot_expense_projection(self):
        """Red line shows expected expense trajectory with shaded confidence bounds."""
        # Smooth bounds for confidence interval
        years_smooth = np.linspace(self.years.min(), self.years.max(), 100)
        high_spline_expense = make_interp_spline(self.years, self.expense_high, k=3)(years_smooth)
        low_spline_expense = make_interp_spline(self.years, self.expense_low, k=3)(years_smooth)

        plt.figure(figsize=(10, 6))

        # Plot median expense
        plt.plot(self.years, self.expense_median, marker='o', linestyle='-', color=self.colors["Total Expense"],
                 label="Expected (Median)")

        # Fill area for upper and lower bounds
        plt.fill_between(years_smooth, low_spline_expense, high_spline_expense, color=self.colors["Total Expense"],
                         alpha=0.3, label="Projection Range")

        plt.xlabel("Year")
        plt.ylabel("Total Expense (¥M)")
        plt.title("Expense Projection")
        plt.legend()
        plt.grid(True)
        plt.xticks(self.years)

        if self.plot:
            plt.show()
        self.save_plot("expense_projection")

    def plot_expense_projection_log(self):
        """Red line represents expected total expenses, now shown on a log scale."""
        # Smooth bounds for confidence interval
        years_smooth = np.linspace(self.years.min(), self.years.max(), 100)
        high_spline_expense = make_interp_spline(self.years, self.expense_high, k=3)(years_smooth)
        low_spline_expense = make_interp_spline(self.years, self.expense_low, k=3)(years_smooth)

        plt.figure(figsize=(10, 6))

        # Plot median expense
        plt.plot(self.years, self.expense_median, marker='o', linestyle='-', color=self.colors["Total Expense"],
                 label="Expected (Median)")

        # Fill area for upper and lower bounds
        plt.fill_between(years_smooth, low_spline_expense, high_spline_expense, color=self.colors["Total Expense"],
                         alpha=0.3, label="Projection Range")

        plt.xlabel("Year")
        plt.ylabel("Total Expense (¥M)")
        plt.title("Expense Projection (Logarithmic)")
        plt.yscale("log")
        plt.ylim(10 ** 2, 10 ** 5)
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.6)
        plt.xticks(self.years)

        if self.plot:
            plt.show()
        self.save_plot("expense_projection_log")

    def plot_financial_overview(self):
        """Revenue, Expense, and Profit plotted together with confidence bounds."""
        years_smooth = np.linspace(self.years.min(), self.years.max(), 100)
        high_spline_revenue = make_interp_spline(self.years, self.revenue_high, k=3)(years_smooth)
        low_spline_revenue = make_interp_spline(self.years, self.revenue_low, k=3)(years_smooth)
        high_spline_expense = make_interp_spline(self.years, self.expense_high, k=3)(years_smooth)
        low_spline_expense = make_interp_spline(self.years, self.expense_low, k=3)(years_smooth)
        high_spline_profit = make_interp_spline(self.years, self.profit_high, k=3)(years_smooth)
        low_spline_profit = make_interp_spline(self.years, self.profit_low, k=3)(years_smooth)

        plt.figure(figsize=(10, 6))

        # Plot median values
        plt.plot(self.years, self.revenue_median, marker='o', linestyle='-', color=self.colors["Total Revenue"],
                 label="Total Revenue (Median)")
        plt.plot(self.years, self.expense_median, marker='o', linestyle='-', color=self.colors["Total Expense"],
                 label="Total Expense (Median)")
        plt.plot(self.years, self.profit_median, marker='o', linestyle='-', color=self.colors["Profit"],
                 label="Profit (Median)")

        # Fill confidence bounds
        plt.fill_between(years_smooth, low_spline_revenue, high_spline_revenue, color=self.colors["Total Revenue"],
                         alpha=0.3, label="Revenue Range")
        plt.fill_between(years_smooth, low_spline_expense, high_spline_expense, color=self.colors["Total Expense"],
                         alpha=0.3, label="Expense Range")
        plt.fill_between(years_smooth, low_spline_profit, high_spline_profit, color=self.colors["Profit"], alpha=0.3,
                         label="Profit Range")

        plt.xlabel("Year")
        plt.ylabel("Amount (¥M)")
        plt.title("Financial Overview")
        plt.legend()
        plt.grid(True)
        plt.xticks(self.years)

        if self.plot:
            plt.show()
        self.save_plot("financial_overview")

    def plot_financial_overview_log(self):
        """Revenue, Expense, and Profit plotted together with confidence bounds on a log scale."""
        years_smooth = np.linspace(self.years.min(), self.years.max(), 100)
        high_spline_revenue = make_interp_spline(self.years, self.revenue_high, k=3)(years_smooth)
        low_spline_revenue = make_interp_spline(self.years, self.revenue_low, k=3)(years_smooth)
        high_spline_expense = make_interp_spline(self.years, self.expense_high, k=3)(years_smooth)
        low_spline_expense = make_interp_spline(self.years, self.expense_low, k=3)(years_smooth)
        high_spline_profit = make_interp_spline(self.years, self.profit_high, k=3)(years_smooth)
        low_spline_profit = make_interp_spline(self.years, self.profit_low, k=3)(years_smooth)

        plt.figure(figsize=(10, 6))

        # Plot median values
        plt.plot(self.years, self.revenue_median, marker='o', linestyle='-', color=self.colors["Total Revenue"],
                 label="Total Revenue (Median)")
        plt.plot(self.years, self.expense_median, marker='o', linestyle='-', color=self.colors["Total Expense"],
                 label="Total Expense (Median)")
        plt.plot(self.years, self.profit_median, marker='o', linestyle='-', color=self.colors["Profit"],
                 label="Profit (Median)")

        # Fill confidence bounds
        plt.fill_between(years_smooth, low_spline_revenue, high_spline_revenue, color=self.colors["Total Revenue"],
                         alpha=0.3, label="Revenue Range")
        plt.fill_between(years_smooth, low_spline_expense, high_spline_expense, color=self.colors["Total Expense"],
                         alpha=0.3, label="Expense Range")
        plt.fill_between(years_smooth, low_spline_profit, high_spline_profit, color=self.colors["Profit"], alpha=0.3,
                         label="Profit Range")

        plt.xlabel("Year")
        plt.ylabel("Amount (¥M)")
        plt.title("Financial Overview (Logarithmic)")
        plt.yscale("log")
        plt.ylim(10 ** 2, 10 ** 5)
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.6)
        plt.xticks(self.years)

        if self.plot:
            plt.show()
        self.save_plot("financial_overview_log")

    def plot_growth_rate(self):
        """Line chart showing revenue, expense, and profit growth rates from Year 2 onward."""
        revenue_growth = np.diff(self.revenue_median, prepend=self.revenue_median[0]) / self.revenue_median * 100
        expense_growth = np.diff(self.expense_median, prepend=self.expense_median[0]) / self.expense_median * 100
        profit_growth = np.diff(self.profit_median, prepend=self.profit_median[0]) / self.profit_median * 100

        plt.figure(figsize=(10, 6))

        # Plot all three growth rates from Year 2 onward
        plt.plot(self.years[1:], revenue_growth[1:], marker='o', linestyle='-', color=self.colors["Total Revenue"],
                 label="Revenue Growth Rate")
        plt.plot(self.years[1:], expense_growth[1:], marker='o', linestyle='-', color=self.colors["Total Expense"],
                 label="Expense Growth Rate")
        plt.plot(self.years[1:], profit_growth[1:], marker='o', linestyle='-', color=self.colors["Profit"],
                 label="Profit Growth Rate")

        plt.xlabel("Year")
        plt.ylabel("Growth Rate (%)")
        plt.title("Revenue, Expense, and Profit Growth Rates")
        plt.legend()
        plt.grid(True)
        plt.xticks(self.years[1:])

        if self.plot:
            plt.show()
        self.save_plot("growth_rate")

    def plot_membership_tiers(self):
        """
        For each year, draw two stacked bars:
          - Left bar: revenue by tier (¥M) [left y-axis]
          - Right bar: member counts by tier (schools) [right y-axis]
        Plus two lines:
          - Total revenue (left y-axis)
          - Total member count (right y-axis)
        """

        # Revenue data (¥M)
        rev_basic = self.revenue_basic
        rev_adv = self.revenue_advanced
        rev_prem = self.revenue_premium
        rev_total = self.revenue_memberships

        # Member counts (schools)
        mem_basic = self.members_basic
        mem_adv = self.members_advanced
        mem_prem = self.members_premium
        mem_total = [b + a + p for b, a, p in zip(mem_basic, mem_adv, mem_prem)]

        # Layout
        x = np.arange(len(self.years))
        bar_width = 0.35
        offset = bar_width / 2

        fig, ax_rev = plt.subplots(figsize=(12, 6))
        ax_cnt = ax_rev.twinx()

        # Left stacked bar (revenue) at x - offset
        rev_x = x - offset
        ax_rev.bar(
            rev_x, rev_basic, width=bar_width,
            color=self.colors.get("Basic Membership"), alpha=0.85,
            label="Basic Revenue (¥M)"
        )
        ax_rev.bar(
            rev_x, rev_adv, width=bar_width, bottom=rev_basic,
            color=self.colors.get("Advanced Membership"), alpha=0.85,
            label="Advanced Revenue (¥M)"
        )
        ax_rev.bar(
            rev_x, rev_prem, width=bar_width, bottom=[b + a for b, a in zip(rev_basic, rev_adv)],
            color=self.colors.get("Premium Membership"), alpha=0.85,
            label="Premium Revenue (¥M)"
        )

        # Right stacked bar (counts) at x + offset
        cnt_x = x + offset
        ax_cnt.bar(
            cnt_x, mem_basic, width=bar_width,
            color=self.colors.get("Basic Membership"), alpha=0.45,
            label="Basic (schools)"
        )
        ax_cnt.bar(
            cnt_x, mem_adv, width=bar_width, bottom=mem_basic,
            color=self.colors.get("Advanced Membership"), alpha=0.45,
            label="Advanced (schools)"
        )
        ax_cnt.bar(
            cnt_x, mem_prem, width=bar_width, bottom=[b + a for b, a in zip(mem_basic, mem_adv)],
            color=self.colors.get("Premium Membership"), alpha=0.45,
            label="Premium (schools)"
        )

        # Lines: totals
        ax_rev.plot(
            rev_x, rev_total, marker="o", linestyle="-",
            color=self.colors.get("Membership Revenue", "C0"),
            label="Total Membership Revenue (¥M)"
        )
        ax_cnt.plot(
            cnt_x, mem_total, marker="o", linestyle="-",
            color=self.colors.get("Total Members", "C1"),
            label="Total Member Institutions"
        )

        # Axis labels, limits, grid
        ax_rev.set_xlabel("Year")
        ax_rev.set_ylabel("Revenue (¥M)")
        ax_cnt.set_ylabel("Member Institutions (count)")

        ax_rev.set_ylim(0, max((rev_basic + rev_adv + rev_prem).max(), rev_total.max()) * 1.15)
        ax_cnt.set_ylim(0, max(mem_total) * 1.2)

        ax_rev.set_xticks(
            x,
            labels=[f"Year {int(y)}" if isinstance(y, (int, np.integer)) else str(y) for y in self.years]
        )
        ax_rev.grid(True, axis="y", linestyle="--", alpha=0.35)

        ax_rev.set_title("Membership Revenue (¥M) and Member Counts by Tier")

        # Combined legend
        h1, l1 = ax_rev.get_legend_handles_labels()
        h2, l2 = ax_cnt.get_legend_handles_labels()
        ax_rev.legend(h1 + h2, l1 + l2, loc="upper left", ncol=2, frameon=False)

        fig.tight_layout()
        if self.plot:
            plt.show()
        self.save_plot("membership_tiers")

    def plot_profit_margin(self):
        """Black line shows expected profit margin, with smooth confidence bounds."""
        profit_margin = (self.profit_median / self.revenue_median) * 100
        profit_margin_high = (self.profit_high / self.revenue_high) * 100
        profit_margin_low = (self.profit_low / self.revenue_low) * 100

        # Create a smoother range of years
        years_smooth = np.linspace(self.years.min(), self.years.max(), 100)

        # Apply B-spline interpolation to smooth only the bounds
        high_spline = make_interp_spline(self.years, profit_margin_high, k=3)(years_smooth)
        low_spline = make_interp_spline(self.years, profit_margin_low, k=3)(years_smooth)

        plt.figure(figsize=(10, 6))

        # Plot median profit margin (unsmoothed)
        plt.plot(self.years, profit_margin, marker='o', linestyle='-', color=self.colors["Profit Margin"],
                 label="Profit Margin (Median)")

        # Fill confidence bounds with smoothed interpolation
        plt.fill_between(years_smooth, low_spline, high_spline,
                         color=self.colors["Profit Margin"], alpha=0.3, label="Profit Margin Range")

        plt.xlabel("Year")
        plt.ylabel("Profit Margin (%)")
        plt.title("Profit Margin Over Time")
        plt.legend()
        plt.grid(True)
        plt.xticks(self.years)

        if self.plot:
            plt.show()
        self.save_plot("profit_margin")

    def plot_profit_projection(self):
        """Green line represents expected (median) profit trajectory with shaded confidence bounds."""
        # Smooth bounds for confidence interval
        years_smooth = np.linspace(self.years.min(), self.years.max(), 100)
        high_spline_profit = make_interp_spline(self.years, self.profit_high, k=3)(years_smooth)
        low_spline_profit = make_interp_spline(self.years, self.profit_low, k=3)(years_smooth)

        plt.figure(figsize=(10, 6))

        # Plot median profit
        plt.plot(self.years, self.profit_median, marker='o', linestyle='-', color=self.colors["Profit"],
                 label="Expected (Median)")

        # Fill area for upper and lower bounds
        plt.fill_between(years_smooth, low_spline_profit, high_spline_profit, color=self.colors["Profit"], alpha=0.3,
                         label="Projection Range")

        plt.xlabel("Year")
        plt.ylabel("Total Profit (¥M)")
        plt.title("Profit Projection")
        plt.legend()
        plt.grid(True)
        plt.xticks(self.years)

        if self.plot:
            plt.show()
        self.save_plot("profit_projection")

    def plot_revenue_breakdown(self):
        """Stacked bar chart showing how revenue is allocated across different sources."""
        membership = self.revenue_memberships
        sponsorship = self.revenue_sponsorships
        grants = self.revenue_grants
        licensing = self.revenue_licensing
        venture = self.revenue_ventures

        # Compute the stacked bottom positions dynamically
        bottom_sponsorship = membership
        bottom_grants = bottom_sponsorship + sponsorship
        bottom_licensing = bottom_grants + grants
        bottom_venture = bottom_licensing + licensing

        plt.figure(figsize=(10, 6))

        plt.bar(self.years, membership, label="Membership", color=self.colors["Membership Revenue"])
        plt.bar(self.years, sponsorship, bottom=bottom_sponsorship, label="Sponsorship",
                color=self.colors["Sponsorship Revenue"])
        plt.bar(self.years, grants, bottom=bottom_grants, label="Grants", color=self.colors["Grants Revenue"])
        plt.bar(self.years, licensing, bottom=bottom_licensing, label="Licensing",
                color=self.colors["Licensing Revenue"])
        plt.bar(self.years, venture, bottom=bottom_venture, label="Venture", color=self.colors["Venture Revenue"])

        plt.xlabel("Year")
        plt.ylabel("Revenue Allocation (¥M)")
        plt.title("Revenue Breakdown By Source")
        plt.xticks(self.years)
        plt.legend()
        plt.grid(True, axis='y')

        if self.plot:
            plt.show()
        self.save_plot("revenue_breakdown")

    def plot_revenue_breakdown_proportional(self):
        """Normalized stacked bar chart showing revenue allocation as percentages."""
        total_revenue = (
                self.revenue_memberships +
                self.revenue_sponsorships +
                self.revenue_grants +
                self.revenue_licensing +
                self.revenue_ventures
        )

        plt.figure(figsize=(10, 6))

        bottom = np.zeros_like(self.revenue_memberships, dtype=float)

        plt.bar(self.years, (self.revenue_memberships / total_revenue) * 100, bottom=bottom,
                label="Membership", color=self.colors["Membership Revenue"])
        bottom += (self.revenue_memberships / total_revenue) * 100

        plt.bar(self.years, (self.revenue_sponsorships / total_revenue) * 100, bottom=bottom,
                label="Sponsorship", color=self.colors["Sponsorship Revenue"])
        bottom += (self.revenue_sponsorships / total_revenue) * 100

        plt.bar(self.years, (self.revenue_grants / total_revenue) * 100, bottom=bottom,
                label="Grants", color=self.colors["Grants Revenue"])
        bottom += (self.revenue_grants / total_revenue) * 100

        plt.bar(self.years, (self.revenue_licensing / total_revenue) * 100, bottom=bottom,
                label="Licensing", color=self.colors["Licensing Revenue"])
        bottom += (self.revenue_licensing / total_revenue) * 100

        plt.bar(self.years, (self.revenue_ventures / total_revenue) * 100, bottom=bottom,
                label="Venture", color=self.colors["Venture Revenue"])

        plt.xlabel("Year")
        plt.ylabel("Revenue Allocation (%)")
        plt.title("Revenue Breakdown By Source (Proportional)")
        plt.xticks(self.years)
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(True, axis='y')

        if self.plot:
            plt.show()
        self.save_plot("revenue_breakdown_proportional")

    def plot_revenue_projection(self):
        """Blue line represents expected revenue growth with shaded confidence bounds."""
        # Smooth bounds for confidence interval
        years_smooth = np.linspace(self.years.min(), self.years.max(), 100)
        high_spline_revenue = make_interp_spline(self.years, self.revenue_high, k=3)(years_smooth)
        low_spline_revenue = make_interp_spline(self.years, self.revenue_low, k=3)(years_smooth)

        plt.figure(figsize=(10, 6))

        # Plot median revenue
        plt.plot(self.years, self.revenue_median, marker='o', linestyle='-', color=self.colors["Total Revenue"],
                 label="Expected (Median)")

        # Fill area for upper and lower bounds
        plt.fill_between(years_smooth, low_spline_revenue, high_spline_revenue, color=self.colors["Total Revenue"],
                         alpha=0.3, label="Projection Range")

        plt.xlabel("Year")
        plt.ylabel("Total Revenue (¥M)")
        plt.title("Revenue Projection")
        plt.legend()
        plt.grid(True)
        plt.xticks(self.years)

        if self.plot:
            plt.show()
        self.save_plot("revenue_projection")

    def plot_revenue_projection_log(self):
        """Blue line shows expected revenue growth on a logarithmic scale."""
        # Smooth bounds for confidence interval
        years_smooth = np.linspace(self.years.min(), self.years.max(), 100)
        high_spline_revenue = make_interp_spline(self.years, self.revenue_high, k=3)(years_smooth)
        low_spline_revenue = make_interp_spline(self.years, self.revenue_low, k=3)(years_smooth)

        plt.figure(figsize=(10, 6))

        # Plot median revenue
        plt.plot(self.years, self.revenue_median, marker='o', linestyle='-', color=self.colors["Total Revenue"],
                 label="Expected (Median)")

        # Fill area for upper and lower bounds
        plt.fill_between(years_smooth, low_spline_revenue, high_spline_revenue, color=self.colors["Total Revenue"],
                         alpha=0.3, label="Projection Range")

        plt.xlabel("Year")
        plt.ylabel("Total Revenue (¥M)")
        plt.title("Revenue Projection (Logarithmic)")
        plt.yscale("log")
        plt.ylim(10 ** 2, 10 ** 5)
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.6)
        plt.xticks(self.years)

        if self.plot:
            plt.show()
        self.save_plot("revenue_projection_log")

    def plot_revenue_sources(self):
        """Each revenue source is plotted separately with confidence bounds (±10%)."""
        revenue_sources = {
            "Membership": self.revenue_memberships,
            "Sponsorship": self.revenue_sponsorships,
            "Grants": self.revenue_grants,
            "Licensing": self.revenue_licensing,
            "Venture": self.revenue_ventures
        }

        plt.figure(figsize=(10, 6))

        for source, values in revenue_sources.items():
            upper_bound = values * 1.1  # +10%
            lower_bound = values * 0.9  # -10%
            plt.plot(self.years, values, marker='o', linestyle='-', label=source,
                     color=self.colors[source + " Revenue"])
            plt.fill_between(self.years, lower_bound, upper_bound, color=self.colors[source + " Revenue"], alpha=0.2)

        plt.xlabel("Year")
        plt.ylabel("Revenue (¥M)")
        plt.title("Revenue Sources Over Time (With Bounds)")
        plt.legend()
        plt.grid(True)
        plt.xticks(self.years)

        if self.plot:
            plt.show()
        self.save_plot("revenue_sources")

    def plot_sponsorship_vs_membership(self):
        """Compare sponsorship revenue with membership revenue over time."""
        sponsorship = self.revenue_median - self.num_schools  # Approximate sponsorship contribution
        membership = self.num_schools  # Membership revenue directly

        plt.figure(figsize=(10, 6))

        # Plot sponsorship and membership revenue
        plt.plot(self.years, sponsorship, marker='o', linestyle='-', color=self.colors["Sponsorship Revenue"],
                 label="Sponsorship Revenue")
        plt.plot(self.years, membership, marker='o', linestyle='-', color=self.colors["Membership Revenue"],
                 label="Membership Revenue")

        plt.xlabel("Year")
        plt.ylabel("Revenue (¥M)")
        plt.title("Partnership vs Membership Revenue")
        plt.legend()
        plt.grid(True)
        plt.xticks(self.years)

        if self.plot:
            plt.show()
        self.save_plot("sponsorship_vs_membership")

    def plot_all(self, show_plot=False):
        self.plot = show_plot
        self.plot_cost_efficiency()
        self.plot_expense_breakdown()
        self.plot_expense_breakdown_proportional()
        self.plot_expense_projection()
        self.plot_expense_projection_log()
        self.plot_financial_overview()
        self.plot_financial_overview_log()
        self.plot_growth_rate()
        self.plot_membership_tiers()
        self.plot_profit_margin()
        self.plot_profit_projection()
        self.plot_revenue_breakdown()
        self.plot_revenue_breakdown_proportional()
        self.plot_revenue_projection()
        self.plot_revenue_projection_log()
        self.plot_revenue_projection_log()
        self.plot_revenue_sources()
        self.plot_sponsorship_vs_membership()

    def save_plot(self, name):
        """Save the current figure as a high-resolution image."""
        file_path = f"plots/{name}.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {file_path}")

# Instantiate the class
finance = FrontiersFinance()
finance.plot_all(show_plot=False)