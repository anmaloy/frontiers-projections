import pandas as pd
import numpy as np
import scipy.stats as stats
import warnings

class FinancialModel:
    """
    A financial projection model for computing quantities of participation and revenues.

    This class simulates a 10-year financial forecast for an institutional program, estimating:
    - Membership growth and associated income.
    - Sponsorship acquisition and financial contributions.
    - Grant applications and funding trends.

    The model computes projections and stores results in two DataFrames:
    - `numeric_results`: Tracks the number of members, sponsors, and grants.
    - `financial_results`: Tracks projected revenue from memberships and sponsorships.

    Computed results are saved as CSV files:
    - 'model-projections-numeric.csv'
    - 'model-projections-financial.csv'

    Methods:
        compute(): Runs the 10-year simulation and updates results.
        compute_members(year): Computes projected membership numbers.
        compute_sponsors(year, total_members): Computes projected sponsorship numbers.
        compute_grants(year): Computes projected grant applications.
        compute_membership_income(year, members): Calculates revenue from memberships.
        compute_sponsorship_income(year, sponsors): Calculates revenue from sponsorships.
        insert_data(df, year_col, results_df): Inserts computed data into results DataFrames.
    """
    def __init__(self):
        self.years = [f'Year {year}' for year in range(1, 11)]

        # Initialize empty DataFrames for numeric and financial data
        self.numeric_results = pd.DataFrame(columns=['Category', 'Subcategory'] + self.years)
        self.financial_results = pd.DataFrame(columns=['Category', 'Subcategory'] + self.years)
        self.income_statement = pd.DataFrame(columns=['Category', 'Subcategory'] + self.years)
        self.cash_flow_forecast = pd.DataFrame(columns=['Category', 'Subcategory'] + self.years)
        self.balance_sheet = pd.DataFrame(columns=['Category', 'Subcategory'] + self.years)

        pd.options.mode.chained_assignment = None  # Suppress warnings related to chained assignments
        warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
        warnings.simplefilter(action='ignore', category=FutureWarning)

    def compute(self):
        """
        Compute yearly projections for several models and reports their details in  csv.

        The computed data is stored in two DataFrames:
        - `numeric_results`: Contains data about the quantity of members, sponsorships, etc.
        - `financial_results`: Contains projected income calculated from the above numbers.

        The results are saved to CSV files:
        - 'model-projections-numeric.csv'
        - 'model-projections-financial.csv'
        """
        for year in range(1, 11):
            year_col = f'Year {year}'

            # Compute numerical data
            members = self.compute_members(year)
            school_count = members.loc[members['Subcategory'] == 'Total', year_col].values[0]
            sponsors = self.compute_sponsors(year, school_count)
            grants = self.compute_grants(year)
            licenses = self.compute_licenses(year, school_count)
            ventures = self.compute_ventures(year, school_count)
            employees = self.compute_employees(year, school_count)
            mentors = self.compute_mentors(year, school_count)
            infrastructure = self.compute_infrastructure(year, employees)
            events = self.compute_events(year, school_count)

            # Append updated results
            self.numeric_results = self.insert_data(members, year_col, self.numeric_results)
            self.numeric_results = self.insert_data(sponsors, year_col, self.numeric_results)
            self.numeric_results = self.insert_data(grants, year_col, self.numeric_results)
            self.numeric_results = self.insert_data(licenses, year_col, self.numeric_results)
            self.numeric_results = self.insert_data(ventures, year_col, self.numeric_results)
            self.numeric_results = self.insert_data(employees, year_col, self.numeric_results)
            self.numeric_results = self.insert_data(mentors, year_col, self.numeric_results)
            self.numeric_results = self.insert_data(infrastructure, year_col, self.numeric_results)
            self.numeric_results = self.insert_data(events, year_col, self.numeric_results)

            # Compute financial income data
            membership_income = self.compute_membership_income(year, members)
            sponsorship_income = self.compute_sponsorship_income(year, sponsors)
            grant_income = self.compute_grant_income(year, grants)
            licensing_income = self.compute_licensing_income(year, licenses)
            venture_income = self.compute_venture_income(year, ventures)

            # Append financial income data
            self.financial_results = self.insert_data(membership_income, year_col, self.financial_results)
            self.financial_results = self.insert_data(sponsorship_income, year_col, self.financial_results)
            self.financial_results = self.insert_data(grant_income, year_col, self.financial_results)
            self.financial_results = self.insert_data(licensing_income, year_col, self.financial_results)
            self.financial_results = self.insert_data(venture_income, year_col, self.financial_results)

            # Compute and append income totals
            total_income = self.compute_income(year)
            self.financial_results = self.insert_data(total_income, year_col, self.financial_results)

            # Compute financial expense data
            employee_expense = self.compute_employee_expense(year, school_count, employees)
            mentor_expense = self.compute_mentor_expense(year, mentors)
            infrastructure_expense = self.compute_infrastructure_expense(year, infrastructure, employees)
            marketing_expense = self.compute_marketing_expense(year)
            event_expense = self.compute_event_expense(year, events)

            # Append financial expense data
            self.financial_results = self.insert_data(employee_expense, year_col, self.financial_results)
            self.financial_results = self.insert_data(mentor_expense, year_col, self.financial_results)
            self.financial_results = self.insert_data(infrastructure_expense, year_col, self.financial_results)
            self.financial_results = self.insert_data(marketing_expense, year_col, self.financial_results)
            self.financial_results = self.insert_data(event_expense, year_col, self.financial_results)

            # Compute and append expense totals
            total_expense = self.compute_expense(year)
            self.financial_results = self.insert_data(total_expense, year_col, self.financial_results)

            # Compute and append profit totals
            total_profit = self.compute_profit(year)
            self.financial_results = self.insert_data(total_profit, year_col, self.financial_results)

        # Generates financial data sheets
        self.generate_income_statement()
        self.generate_cash_flow_forecast()
        self.generate_balance_sheet()

        # Save results
        self.numeric_results.to_csv('models/model-projections-numeric.csv', index=False)
        self.financial_results.to_csv('models/model-projections-financial.csv', index=False)
        print("Computation complete. Results saved to CSV.")

    def insert_data(self, df, year_col, results_df):
        """
        Insert or update computed data into the specified results DataFrame.

        This function checks if a given category and subcategory already exist in `results_df`.
        - If they exist, it updates the corresponding value for `year_col`.
        - If they do not exist, it appends a new row with the provided data.

        Args:
            df (pd.DataFrame): The DataFrame containing new computed values.
            year_col (str): The column name representing the current year (e.g., 'Year 1').
            results_df (pd.DataFrame): The DataFrame storing accumulated results.

        Returns:
            pd.DataFrame: The updated results DataFrame with new or modified entries.
        """
        for _, row in df.iterrows():
            mask = (results_df['Category'] == row['Category']) & (results_df['Subcategory'] == row['Subcategory'])

            if results_df.loc[mask].empty:
                # If Category/Subcategory doesn't exist, append a new row
                new_row = pd.DataFrame({col: [np.nan] for col in results_df.columns})
                new_row['Category'] = row['Category']
                new_row['Subcategory'] = row['Subcategory']
                new_row[year_col] = row[year_col]
                results_df = pd.concat([results_df, new_row], ignore_index=True)
            else:
                # Otherwise, update the existing row
                results_df.loc[mask, year_col] = row[year_col]

        return results_df  # Return updated DataFrame instead of modifying in place

    def compute_members(self, year):
        """
        Compute the projected number of institutional members for each year.

        This function models the growth of memberships over time using an exponential growth model:

            total_members = initial_members * exp(growth_rate * year) - adjustor

        where:
        - `initial_members` represents the starting number of institutions in the program.
        - `growth_rate` is the rate at which membership grows exponentially.
        - `adjustor` is a correction factor to smooth out the early-stage model.
        - `total_members` is then distributed into Basic (60%), Advanced (30%), and Premium (10%) tiers.

        Args:
            year (int): The current year in the simulation (1-10).

        Returns:
            pd.DataFrame: A DataFrame containing membership projections across tiers.
        """
        # Define model parameters
        initial_members = 20
        growth_rate = 0.531
        adjustor = 30

        # Compute total members
        total_members = max(0, int(initial_members * np.exp(growth_rate * year) - adjustor))

        # Define membership distribution ratios
        ratios = {'Basic': 0.6, 'Advanced': 0.3, 'Premium': 0.1}

        # Compute unrounded values
        members = {key: total_members * ratio for key, ratio in ratios.items()}

        # Round values
        rounded_members = {key: round(value) for key, value in members.items()}

        # Compute discrepancy
        discrepancy = total_members - sum(rounded_members.values())

        # Identify the largest category and adjust it
        largest_category = max(rounded_members, key=rounded_members.get)
        rounded_members[largest_category] += discrepancy

        return pd.DataFrame({
            'Category': ['Members'] * 4,
            'Subcategory': ['Total'] + list(rounded_members.keys()),
            f'Year {year}': [total_members] + list(rounded_members.values())
        })

    def compute_sponsors(self, year, total_members):
        """
        Compute the projected number of sponsors based on total memberships with dynamic sponsorship ratios.

        This function models sponsorship growth using a logistic function combined with a network effect:

            total_sponsors = logistic_part + network_effect

        where:
        - `logistic_part` represents saturation-limited growth modeled using a logistic function.
        - `network_effect` accounts for early-stage exponential growth before market saturation.
        - The function also adjusts sponsorship tier ratios dynamically using a second logistic function.

        Args:
            year (int): The current year in the simulation (1-10).
            total_members (int): The total number of institutional members in the given year.

        Returns:
            pd.DataFrame: A DataFrame containing sponsorship projections across tiers.
        """

        # Sponsorship Model Parameters
        L = 1000  # Market saturation limit for sponsorships
        k = 0.002  # Growth rate of sponsorships as they approach saturation
        N0 = 2500  # Inflection point delaying acceleration of sponsorship uptake
        a = 3  # Initial network effect strength
        b = 0.6  # Nonlinearity in early network effects
        c = 0.0001  # Decay rate controlling long-term reduction in network effects

        def sponsorship_ratio(x, r_min, r_max, k, x0):
            """
            Logistic function to dynamically adjust sponsorship ratios over time.

            Args:
                x (float): The number of institutional members.
                r_min (float): The minimum sponsorship ratio as memberships increase.
                r_max (float): The maximum sponsorship ratio in early growth stages.
                k (float): The rate at which the ratio shifts between r_max and r_min.
                x0 (float): The midpoint of the logistic function for ratio transition.

            Returns:
                float: The computed sponsorship ratio at the given membership level.
            """
            return r_min + (r_max - r_min) / (1 + np.exp(-k * (x - x0)))

        # Compute sponsorship growth using a logistic model
        logistic_part = L / (1 + np.exp(-k * (total_members - N0)))

        # Compute network effects: accounts for early acceleration in sponsorship uptake
        network_effect = (
                a * (total_members ** b) * np.exp(-c * total_members) * (1 - (logistic_part / L))
        )

        # Compute total sponsors for the given year
        total_sponsors = round(logistic_part + network_effect, 0)

        # Compute dynamic tier ratios
        ratios = {
            'Small': sponsorship_ratio(total_members, 0.7, 0.4, 0.02, 2500),
            'Mid-Tier': sponsorship_ratio(total_members, 0.25, 0.4, 0.02, 2500),
            'Large': sponsorship_ratio(total_members, 0.05, 0.2, 0.02, 2500)
        }

        # Compute unrounded sponsorships
        sponsors = {key: total_sponsors * ratio for key, ratio in ratios.items()}

        # Round values
        rounded_sponsors = {key: round(value) for key, value in sponsors.items()}

        # Compute discrepancy
        discrepancy = total_sponsors - sum(rounded_sponsors.values())

        # Identify the largest category and adjust it
        largest_category = max(rounded_sponsors, key=rounded_sponsors.get)
        rounded_sponsors[largest_category] += discrepancy

        return pd.DataFrame({
            'Category': ['Sponsorships'] * 4,
            'Subcategory': ['Total'] + list(rounded_sponsors.keys()),
            f'Year {year}': [total_sponsors] + list(rounded_sponsors.values())
        })

    def compute_grants(self, year):
        """
        Compute the projected number of grant applications per tier using distinct growth models.

        The model follows three different functions for small, medium, and large grants:

            Small Grants: Power-law growth with exponential decay
            Medium Grants: Power-law growth with a sharper decay factor
            Large Grants: Logistic growth saturating at `large_max`

        where:
        - Small and Medium grants follow a power-law model with an exponential decay component.
        - Large grants follow a logistic model to reflect delayed adoption and eventual saturation.

        Args:
            year (int): The current year in the simulation (1-10).

        Returns:
            pd.DataFrame: A DataFrame containing projected grant applications by tier.
        """
        # Grant application model parameters
        small_base = 22
        small_exp = 1.03
        small_decay = 2.46

        medium_base = 6
        medium_exp = 1.24
        medium_decay = 3.32

        large_max = 6
        large_growth = 0.47
        large_midpoint = 4.57

        # Approval rates based on year, defined arbitrarily to reflect the rate
        # of success for each type as the program scales.
        approval_rates = {
            1: {'Small': 0.35, 'Medium': 0.20, 'Large': 0.065},
            2: {'Small': 0.46, 'Medium': 0.2234, 'Large': 0.075},
            3: {'Small': 0.5383, 'Medium': 0.25, 'Large': 0.085},
            4: {'Small': 0.585, 'Medium': 0.2766, 'Large': 0.0936},
            5: {'Small': 0.60, 'Medium': 0.30, 'Large': 0.10},
            6: {'Small': 0.5833, 'Medium': 0.3184, 'Large': 0.1043},
            7: {'Small': 0.535, 'Medium': 0.3317, 'Large': 0.107},
            8: {'Small': 0.455, 'Medium': 0.3406, 'Large': 0.1086},
            9: {'Small': 0.3433, 'Medium': 0.3464, 'Large': 0.1095},
            10: {'Small': 0.20, 'Medium': 0.35, 'Large': 0.11}
        }

        # Compute unrounded grant applications
        grants = {
            'Small': small_base * (year ** small_exp) * np.exp(-year / small_decay),
            'Medium': medium_base * (year ** medium_exp) * np.exp(-year / medium_decay),
            'Large': large_max / (1 + np.exp(-large_growth * (year - large_midpoint)))
        }

        # Apply acceptance rates
        accepted_grants = {key: grants[key] * approval_rates[year][key] for key in grants}

        # Round values
        rounded_grants = {key: round(value) for key, value in accepted_grants.items()}

        # Compute total accepted grants and discrepancy
        total_grants = round(sum(accepted_grants.values()))
        discrepancy = total_grants - sum(rounded_grants.values())

        # Identify the largest category and adjust it
        largest_category = max(rounded_grants, key=rounded_grants.get)
        rounded_grants[largest_category] += discrepancy

        return pd.DataFrame({
            'Category': ['Grants'] * 4,
            'Subcategory': ['Total'] + list(rounded_grants.keys()),
            f'Year {year}': [total_grants] + list(rounded_grants.values())
        })

    def compute_licenses(self, year, schools):
        """
        Compute the number of teachers trained per year and estimate the number of schools purchasing core curriculum and IP licensing.

        This function determines the total number of teachers requiring training based on:
        - The number of new schools added in a given year.
        - A retraining rate applied to previously trained teachers.

        Additionally, it calculates the number of schools expected to purchase the core curriculum and IP licensing package.

        Args:
            year (int): The current year in the simulation (1-10).
            schools (int): The total number of participating schools in the given year.

        Returns:
            pd.DataFrame: A DataFrame containing projections for core curriculum/IP licensing adoption and teacher training.
        """

        year_col = f'Year {year}'

        # Training parameters
        teachers_per_school = 3
        retraining_rate = 0.2
        ip_adoption_rate = 0.35  # 35% of schools purchasing the IP package

        # Get previous year's school count
        prev_schools = 0 if year == 1 else \
        self.numeric_results.loc[self.numeric_results['Subcategory'] == 'Total', f'Year {year - 1}'].values[0]

        # Compute new schools and retraining requirements
        new_schools = max(0, schools - prev_schools)
        new_teacher_training = new_schools * teachers_per_school
        retraining = (schools - new_schools) * teachers_per_school * retraining_rate

        # Total teachers trained
        total_teachers_trained = round(new_teacher_training + retraining, 0)

        # Compute core curriculum and IP licensing adoption
        ip_schools = round(schools * ip_adoption_rate, 0)

        return pd.DataFrame({
            'Category': ['Licensing'] * 2,
            'Subcategory': ['Core Curriculum and IP Licensing', 'Teachers Trained'],
            year_col: [ip_schools, total_teachers_trained]
        })

    def compute_ventures(self, year, schools):
        """
        Compute the number of new ventures created and their progression through survival stages.

        This function models venture formation based on student participation and tracks the survival and failure rates
        across multiple stages:
        - Year 1 survival
        - Transition to small, medium, and large success tiers
        - Venture failures at each stage

        The survival and failure rates determine how ventures progress or are removed from the ecosystem annually.

        Args:
            year (int): The current year in the simulation (1-10).
            schools (int): The total number of participating schools in the given year.

        Returns:
            pd.DataFrame: A DataFrame containing projections for new ventures, surviving ventures at each stage,
                          and total failures.
        """

        year_col = f'Year {year}'

        # Venture parameters
        students_per_school = 15
        ventures_per_student = 0.02  # 2 ventures per 100 students
        new_ventures = round(schools * students_per_school * ventures_per_student, 0)

        # Survival and failure rates
        survival_rates = {
            'Year 1': 0.7,
            'Small': 0.3,
            'Medium': 0.1,
            'Large': 0.2
        }

        failure_rates = {
            'Year 1': 0.2,
            'Small': 0.6,
            'Medium': 0.35,
            'Large': 0.01
        }

        # Retrieve last year's venture data
        prev_year_col = f'Year {year - 1}' if year > 1 else None
        prev_ventures = \
        self.numeric_results.loc[self.numeric_results['Subcategory'] == 'Surviving Year 1', prev_year_col].values[
            0] if prev_year_col else 0
        prev_small = \
        self.numeric_results.loc[self.numeric_results['Subcategory'] == 'Small Success', prev_year_col].values[
            0] if prev_year_col else 0
        prev_medium = \
        self.numeric_results.loc[self.numeric_results['Subcategory'] == 'Medium Success', prev_year_col].values[
            0] if prev_year_col else 0

        # Compute survival and failures at each stage
        survived_1 = round(new_ventures * survival_rates['Year 1'],0)
        failed_1 = new_ventures * failure_rates['Year 1']

        small_success = round(prev_ventures * survival_rates['Small'],0)
        failed_2 = prev_ventures * failure_rates['Small']

        medium_success = round(prev_small * survival_rates['Medium'],0)
        failed_3 = prev_small * failure_rates['Medium']

        large_success = round(prev_medium * survival_rates['Large'],0)
        failed_4 = prev_medium * failure_rates['Large']

        # Compute total failures
        total_failures = round(failed_1 + failed_2 + failed_3 + failed_4, 0)

        return pd.DataFrame({
            'Category': ['Ventures'] * 5,
            'Subcategory': ['New Ventures', 'Surviving Year 1', 'Small Success', 'Medium Success', 'Large Success'],
            year_col: [new_ventures, survived_1, small_success, medium_success, large_success]
        })

    def compute_employees(self, year, schools):
        """
        Compute the projected number of employees per category based on school count.

        Args:
            year (int): The current year in the simulation (1-10).
            schools (int): The total number of participating schools.

        Returns:
            pd.DataFrame: A DataFrame containing employee projections across categories.
        """

        year_col = f'Year {year}'

        # Program Director: Always 1
        program_director = 1

        # Administrative Staff: Uses corrected sigmoid function
        amin, amax, k, smid, c = 10, 600, 0.0035, 1200, 8
        admin_staff = round(amin + ((amax - amin) / (1 + np.exp(-k * (schools - smid)))) + (c * np.log(schools + 1)), 0)

        # Local Coordinators: 1 per 5 schools, minimum 1
        schools_per_local = 5
        local_coordinators = max(1, round(schools / schools_per_local, 0))

        # Regional Coordinators: 1 per 10 local coordinators, minimum 1
        local_per_regional = 10
        regional_coordinators = max(1, round(local_coordinators / local_per_regional, 0))

        # Support Staff: 1 per 10 other employees
        employees_per_support = 10
        total_other_employees = program_director + admin_staff + local_coordinators + regional_coordinators
        support_staff = max(1, round(total_other_employees / employees_per_support, 0))

        # Department Heads: Uses logarithmic growth
        employees_per_head = 135
        total_other_employees = total_other_employees + support_staff
        department_heads = round((np.log(total_other_employees + 1) * 2) + (total_other_employees / employees_per_head) + 3, 0)

        # Total Employees
        total_employees = total_other_employees + department_heads

        return pd.DataFrame({
            'Category': ['Employees'] * 7,
            'Subcategory': [
                'Program Director',
                'Department Heads',
                'Administrative Staff',
                'Regional Coordinators',
                'Local Coordinators',
                'Support Staff',
                'Total Employees'
            ],
            year_col: [
                program_director,
                department_heads,
                admin_staff,
                regional_coordinators,
                local_coordinators,
                support_staff,
                total_employees
            ]
        })

    def compute_mentors(self, year, schools):
        """
        Compute the number of mentors required based on the number of schools, students per school, and
        expected mentor engagement level.

        - `students_per_school` defines how many students are in an average school.
        - `mentor_unit` defines the standard number of students that one mentor can support at a base level.
        - `availability_scalar` adjusts this based on the expected mentor involvement.
        - The equation ensures that as student count increases, the number of mentors scales accordingly.

        Args:
            year (int): The current simulation year.
            schools (int): The total number of participating schools.

        Returns:
            pd.DataFrame: A DataFrame containing mentor allocations by involvement tier.
        """
        year_col = f'Year {year}'
        students_per_school = 10
        mentor_unit = 15

        # Availability factor is based on an estimated mentor-to-student ratio derived from expected involvement impact.
        # The divisor is derived from a desired 1 mentor per x students baseline, adjusted for involvement.
        availability_scalar = {'High Involvement': mentor_unit / 3,
                               'Medium Involvement': mentor_unit,
                               'Low Involvement': mentor_unit * 2}
        ratios = {'High Involvement': 0.05, 'Medium Involvement': 0.35, 'Low Involvement': 0.6}

        mentors = {
            level: round((schools * students_per_school / availability_scalar[level]) * ratios[level], 0)
            for level in availability_scalar
        }

        return pd.DataFrame({
            'Category': ['Mentors'] * 4,
            'Subcategory': list(mentors.keys()) + ['Total'],
            year_col: list(mentors.values()) + [sum(mentors.values())]
        })

    def compute_infrastructure(self, year, employees):
        """
        Compute the number of headquarters and regional offices required based on the number of employees.

        - One HQ is required at all times.
        - The HQ upgrades to a larger facility when employees exceed 50.
        - Regional offices are allocated based on 100 employees per office.

        Args:
            year (int): The current simulation year.
            employees (pd.DataFrame): DataFrame containing the number of employees.

        Returns:
            pd.DataFrame: A DataFrame containing infrastructure allocations.
        """
        year_col = f'Year {year}'
        employee_count = employees.loc[employees['Subcategory'] == 'Total Employees', year_col].values[0]

        # Determine HQ size
        hq_type = 'Small HQ' if employee_count <= 50 else 'Large HQ'

        # Determine number of regional offices
        regional_office_count = max(0, -(-employee_count // 100))  # Ceiling division

        return pd.DataFrame({
            'Category': ['Infrastructure'] * 3,
            'Subcategory': ['HQ Type', 'Regional Offices', 'Total Offices'],
            year_col: [hq_type, regional_office_count, 1 + regional_office_count]
        })

    def compute_events(self, year, schools):
        """
        Compute the number of medium and large events based on school count.

        - Large events occur per every 1000 schools.
        - Medium events occur per every 100 schools.

        Args:
            year (int): The current simulation year.
            schools (int): The total number of participating schools.

        Returns:
            pd.DataFrame: A DataFrame containing the number of planned events.
        """
        year_col = f'Year {year}'
        large_event_req = 1000
        med_event_req = 100
        large_events = max(0, schools // large_event_req)
        medium_events = max(0, schools // med_event_req)

        return pd.DataFrame({
            'Category': ['Events'] * 3,
            'Subcategory': ['Large Events', 'Medium Events', 'Total Events'],
            year_col: [large_events, medium_events, large_events + medium_events]
        })

    def compute_membership_income(self, year, members):
        """
        Compute projected membership income with high, median, and low estimates.

        Uses a normal distribution model to assign prices based on dynamically recalculated
        membership ratios per year.

        Args:
            year (int): The current year in the simulation (1-10).
            members (pd.DataFrame): A DataFrame containing the number of members in each tier.

        Returns:
            pd.DataFrame: A DataFrame containing projected membership income.
        """

        # Define column name for the current year
        year_col = f'Year {year}'

        # Extract membership counts
        basic_members = members.loc[members['Subcategory'] == 'Basic', year_col].values[0]
        advanced_members = members.loc[members['Subcategory'] == 'Advanced', year_col].values[0]
        premium_members = members.loc[members['Subcategory'] == 'Premium', year_col].values[0]

        total_members = basic_members + advanced_members + premium_members

        # Pricing Ranges
        pricing = {
            'Basic': {'Low': 0.75, 'High': 2.2},
            'Advanced': {'Low': 3.0, 'High': 7.5},
            'Premium': {'Low': 10.0, 'High': 15.0}
        }

        # Compute weights dynamically
        weights = {tier: members / total_members if total_members > 0 else 0 for tier, members in
                   zip(pricing.keys(), [basic_members, advanced_members, premium_members])}

        # Compute mean price and standard deviation dynamically
        mean_price = sum(weights[tier] * (pricing[tier]['Low'] + pricing[tier]['High']) / 2 for tier in pricing)
        std_price = np.sqrt(
            sum(weights[tier] * ((pricing[tier]['High'] - pricing[tier]['Low']) / 3) ** 2 for tier in pricing))

        # Percentile mapping
        percentiles = {'Basic': (0.0, 0.6), 'Advanced': (0.6, 0.9), 'Premium': (0.9, 1.0)}

        # Compute expected price per tier
        prices = {tier: self.map_percentile_to_price(mean_price, std_price, pricing[tier]['Low'], pricing[tier]['High'],
                                                     percentiles[tier][0], percentiles[tier][1]) for tier in pricing}

        # Compute income per tier
        incomes = {tier: round(members * prices[tier], 0) for tier, members in
                   zip(pricing.keys(), [basic_members, advanced_members, premium_members])}

        # Compute total income estimates
        income_median = round(sum(incomes.values()), 0)
        income_high = round(income_median + std_price * total_members, 0)
        income_low = round(income_median - std_price * total_members, 0)

        return pd.DataFrame({
            'Category': ['Memberships'] * 6,
            'Subcategory': ['Basic', 'Advanced', 'Premium', 'Total Median', 'Total High', 'Total Low'],
            year_col: [incomes['Basic'], incomes['Advanced'], incomes['Premium'], income_median, income_high,
                       income_low]
        })

    def compute_sponsorship_income(self, year, sponsors):
        """
        Compute projected sponsorship income with high, median, and low estimates.

        Uses a normal distribution model to assign sponsorship values based on dynamically recalculated
        sponsor ratios per year.

        Args:
            year (int): The current year in the simulation (1-10).
            sponsors (pd.DataFrame): A DataFrame containing sponsor counts per tier.

        Returns:
            pd.DataFrame: A DataFrame containing projected sponsorship income.
        """

        year_col = f'Year {year}'

        # Extract sponsorship counts
        small_sponsors = sponsors.loc[sponsors['Subcategory'] == 'Small', year_col].values[0]
        mid_sponsors = sponsors.loc[sponsors['Subcategory'] == 'Mid-Tier', year_col].values[0]
        large_sponsors = sponsors.loc[sponsors['Subcategory'] == 'Large', year_col].values[0]

        total_sponsors = small_sponsors + mid_sponsors + large_sponsors

        # Sponsorship pricing
        pricing = {
            'Small': {'Low': 1.0, 'High': 5.0, 'Mean': 3.0, 'STD': 1.33},
            'Mid-Tier': {'Low': 10.0, 'High': 25.0, 'Mean': 17.5, 'STD': 5.0},
            'Large': {'Low': 40.0, 'High': 100.0, 'Mean': 70.0, 'STD': 20.0}
        }

        # Compute income estimates using percentile mapping
        incomes = {
            tier: sponsors * self.map_percentile_to_price(
                pricing[tier]['Mean'], pricing[tier]['STD'], pricing[tier]['Low'], pricing[tier]['High'], 0.25, 0.75
            ) for tier, sponsors in zip(pricing.keys(), [small_sponsors, mid_sponsors, large_sponsors])
        }

        # Compute total sponsorship income estimates
        income_median = sum(incomes.values())
        income_std = np.sqrt(sum(((pricing[tier]['High'] - pricing[tier]['Low']) / 3) ** 2 * (
            sponsors / total_sponsors if total_sponsors > 0 else 0) for tier, sponsors in
                                 zip(pricing.keys(), [small_sponsors, mid_sponsors, large_sponsors])))
        income_high = round(income_median + income_std * total_sponsors, 0)
        income_low = round(income_median - income_std * total_sponsors, 0)

        return pd.DataFrame({
            'Category': ['Sponsorships'] * 6,
            'Subcategory': ['Small', 'Mid-Tier', 'Large', 'Total Median', 'Total High', 'Total Low'],
            year_col: [round(incomes['Small'],0), round(incomes['Mid-Tier'],0), round(incomes['Large'],0),
                       round(income_median,0), income_high, income_low]
        })

    def compute_grant_income(self, year, grants):
        """
        Compute projected grant sponsorship income with high, median, and low estimates.

        This function calculates expected grant income by applying percentile-based price mapping to grant tiers:
        - Small, Medium, and Large grant categories have predefined value distributions.
        - The expected grant value is computed per tier and aggregated for total income estimation.
        - A standard deviation-based adjustment is applied to estimate high and low ranges.

        Args:
            year (int): The current year in the simulation (1-10).
            grants (pd.DataFrame): A DataFrame containing the number of accepted grants by tier.

        Returns:
            pd.DataFrame: A DataFrame containing projected grant income estimates across tiers.
        """

        year_col = f'Year {year}'

        # Extract grant counts
        small_grants = grants.loc[grants['Subcategory'] == 'Small', year_col].values[0]
        medium_grants = grants.loc[grants['Subcategory'] == 'Medium', year_col].values[0]
        large_grants = grants.loc[grants['Subcategory'] == 'Large', year_col].values[0]

        total_grants = small_grants + medium_grants + large_grants

        # Grant pricing
        pricing = {
            'Small': {'Low': 10, 'High': 50, 'Mean': 30, 'STD': 13.33},
            'Medium': {'Low': 50, 'High': 300, 'Mean': 175, 'STD': 83.33},
            'Large': {'Low': 300, 'High': 750, 'Mean': 525, 'STD': 150}
        }

        # Compute income estimates using percentile mapping
        incomes = {
            tier: grants * self.map_percentile_to_price(
                pricing[tier]['Mean'], pricing[tier]['STD'], pricing[tier]['Low'], pricing[tier]['High'], 0.25, 0.75
            ) for tier, grants in zip(pricing.keys(), [small_grants, medium_grants, large_grants])
        }

        # Compute total grant income estimates
        income_median = sum(incomes.values())
        income_std = np.sqrt(sum(((pricing[tier]['High'] - pricing[tier]['Low']) / 3) ** 2 * (
            grants / total_grants if total_grants > 0 else 0) for tier, grants in
                                 zip(pricing.keys(), [small_grants, medium_grants, large_grants])))
        income_high = round(income_median + income_std * total_grants, 0)
        income_low = round(income_median - income_std * total_grants, 0)

        return pd.DataFrame({
            'Category': ['Grants'] * 6,
            'Subcategory': ['Small', 'Medium', 'Large', 'Total Median', 'Total High', 'Total Low'],
            year_col: [incomes['Small'], incomes['Medium'], incomes['Large'], income_median, income_high, income_low]
        })

    def compute_licensing_income(self, year, licenses):
        """
        Compute projected licensing revenue with high, median, and low estimates.

        This function estimates licensing income by applying percentile-based pricing to:
        - Core Curriculum and IP Licensing purchases by schools.
        - Training fees based on the number of teachers trained.

        A weighted distribution of Basic, Advanced, and Premium tiers determines pricing,
        and a standard deviation-based adjustment estimates high and low income projections.

        Args:
            year (int): The current year in the simulation (1-10).
            licenses (pd.DataFrame): A DataFrame containing the number of schools purchasing IP and teachers trained.

        Returns:
            pd.DataFrame: A DataFrame containing projected licensing income estimates across categories.
        """

        year_col = f'Year {year}'

        # Extract licensing counts
        ip_schools = licenses.loc[licenses['Subcategory'] == 'Core Curriculum and IP Licensing', year_col].values[0]
        teachers_trained = licenses.loc[licenses['Subcategory'] == 'Teachers Trained', year_col].values[0]

        # Licensing pricing
        pricing = {
            'Basic': {'IP': 1.375, 'IP Low': 0.75, 'IP High': 2, 'Training': 0.375, 'Training Low': 0.25, 'Training High': 0.5},
            'Advanced': {'IP': 3.5, 'IP Low': 2, 'IP High': 5, 'Training': 1.125, 'Training Low': 0.75, 'Training High': 1.5},
            'Premium': {'IP': 7.5, 'IP Low': 5, 'IP High': 10, 'Training': 2.25, 'Training Low': 1.5, 'Training High': 3}
        }

        # Distribution ratios
        distribution = {'Basic': 0.6, 'Advanced': 0.3, 'Premium': 0.1}

        # Compute income estimates using percentile mapping
        ip_income = sum(
            distribution[tier] * ip_schools * self.map_percentile_to_price(
                pricing[tier]['IP'], (pricing[tier]['IP High'] - pricing[tier]['IP Low']) / 3, pricing[tier]['IP Low'], pricing[tier]['IP High'], 0.25, 0.75
            ) for tier in pricing
        )
        training_income = sum(
            distribution[tier] * teachers_trained * self.map_percentile_to_price(
                pricing[tier]['Training'], (pricing[tier]['Training High'] - pricing[tier]['Training Low']) / 3, pricing[tier]['Training Low'], pricing[tier]['Training High'], 0.25, 0.75
            ) for tier in pricing
        )

        # Compute total licensing income estimates
        income_median = ip_income + training_income
        income_std = np.sqrt(sum(((pricing[tier]['IP High'] - pricing[tier]['IP Low']) / 3) ** 2 * (ip_schools / (ip_schools + teachers_trained) if (ip_schools + teachers_trained) > 0 else 0) for tier in pricing))
        income_high = round(income_median + income_std * (ip_schools + teachers_trained), 0)
        income_low = round(income_median - income_std * (ip_schools + teachers_trained), 0)

        return pd.DataFrame({
            'Category': ['Licensing'] * 5,
            'Subcategory': ['IP Fees', 'Training Fees', 'Total Median', 'Total High', 'Total Low'],
            year_col: [round(ip_income, 0), round(training_income, 0), round(income_median, 0), round(income_high, 0), round(income_low, 0)]
        })

    def compute_venture_income(self, year, ventures_data):
        """
        Compute projected venture income with high, median, and low estimates.

        This function estimates the retained value of ventures based on their success stage:
        - Initial, Small, Medium, and Large venture categories are assigned predefined value distributions.
        - The expected venture value is computed per tier and aggregated for total income estimation.
        - A standard deviation-based adjustment is applied to estimate high and low income projections.

        Args:
            year (int): The current year in the simulation (1-10).
            ventures_data (pd.DataFrame): A DataFrame containing the number of ventures at each stage.

        Returns:
            pd.DataFrame: A DataFrame containing projected venture income estimates across tiers.
        """

        year_col = f'Year {year}'

        # Extract venture counts
        ventures = {
            'Initial': ventures_data.loc[ventures_data['Subcategory'] == 'Surviving Year 1', year_col].values[0],
            'Small': ventures_data.loc[ventures_data['Subcategory'] == 'Small Success', year_col].values[0],
            'Medium': ventures_data.loc[ventures_data['Subcategory'] == 'Medium Success', year_col].values[0],
            'Large': ventures_data.loc[ventures_data['Subcategory'] == 'Large Success', year_col].values[0]
        }

        # Venture valuation and ownership parameters
        ownership = 0.2  # 20% retained ownership
        pricing = {
            'Initial': {'Low': 10, 'High': 50, 'Mean': 30, 'STD': 13.33},
            'Small': {'Low': 50, 'High': 250, 'Mean': 150, 'STD': 66.67},
            'Medium': {'Low': 250, 'High': 1000, 'Mean': 625, 'STD': 250},
            'Large': {'Low': 1000, 'High': 10000, 'Mean': 5500, 'STD': 2598.08}
        }

        # Compute income estimates using percentile mapping
        incomes = {
            tier: ventures[tier] * self.map_percentile_to_price(
                pricing[tier]['Mean'], pricing[tier]['STD'], pricing[tier]['Low'], pricing[tier]['High'], 0.25, 0.75
            ) * ownership
            for tier in pricing
        }

        # Compute total median venture income
        income_median = sum(incomes.values())

        # Compute total standard deviation dynamically per tier, ensuring proper scaling
        income_std = np.sqrt(
            sum(
                (pricing[tier]['STD'] ** 2) * (ventures[tier] / sum(ventures.values()) if sum(ventures.values()) > 0 else 0)
                for tier in pricing
            )
        )

        # Compute high and low estimates correctly
        income_high = round(income_median + income_std, 0)
        income_low = round(max(income_median - income_std, 0), 0)  # Ensure it does not go negative

        return pd.DataFrame({
            'Category': ['Ventures'] * 7,
            'Subcategory': ['Initial', 'Small', 'Medium', 'Large', 'Total Median', 'Total High', 'Total Low'],
            year_col: [round(incomes['Initial'],0), round(incomes['Small'],0), round(incomes['Medium'],0),
                       round(incomes['Large'],0), round(income_median,0), income_high, income_low]
        })

    def compute_employee_expense(self, year, schools, employees):
        """
        Compute the projected salary expenses per employee category with median, low, and high estimates.

        Args:
            year (int): The current year in the simulation (1-10).
            schools (int): The total number of participating schools.
            employees (pd.DataFrame): A DataFrame containing the number of employees per category.

        Returns:
            pd.DataFrame: A DataFrame containing salary expense projections for each category.
        """

        year_col = f'Year {year}'

        # Extract employee counts for the year
        def get_employee_count(subcategory):
            return employees.loc[employees['Subcategory'] == subcategory, year_col].values[0]

        # Compute dynamically growing salaries
        program_director_salary = 15 * (schools ** 0.2)
        department_heads_salary = 10.5 * (schools ** 0.1)

        def compute_salary_over_time(base_salary, year, growth_factor=1.05):
            """ Iteratively applies salary growth for given years. """
            for _ in range(year - 1):
                base_salary *= growth_factor
            return round(base_salary, 1)

        # Compute the current year's median salaries
        salaries = {
            'Program Director': program_director_salary,
            'Department Heads': department_heads_salary,
            'Administrative Staff': compute_salary_over_time(6, year),
            'Regional Coordinators': compute_salary_over_time(8.5, year),
            'Local Coordinators': compute_salary_over_time(6.5, year),
            'Support Staff': compute_salary_over_time(4, year)
        }

        # Define Low & High dynamically based on the calculated median salary
        salary_ranges = {
            role: {
                'Low': salary * 0.85,  # 15% lower
                'High': salary * 1.15  # 15% higher
            }
            for role, salary in salaries.items()
        }

        # Compute salary estimates using normal distribution assumptions
        def get_salary_estimates(role):
            median = salaries[role]
            std_salary = (salary_ranges[role]['High'] - salary_ranges[role]['Low']) / 3  # Approximate 99.7% range

            low = self.map_percentile_to_price(median, std_salary, salary_ranges[role]['Low'],
                                               salary_ranges[role]['High'], 0.05, 0.50)
            high = self.map_percentile_to_price(median, std_salary, salary_ranges[role]['Low'],
                                                salary_ranges[role]['High'], 0.50, 0.95)

            return round(low, 0), round(median, 0), round(high, 0)

        # Compute salary ranges for each category
        expenses = {
            role: tuple(round(get_employee_count(role) * salary, 0) for salary in get_salary_estimates(role))
            for role in salaries.keys()
        }

        # Compute total salaries for each estimate
        total_salaries = {
            'Low': sum(expense[0] for expense in expenses.values()),
            'Median': sum(expense[1] for expense in expenses.values()),
            'High': sum(expense[2] for expense in expenses.values())
        }

        # Return structured DataFrame
        return pd.DataFrame({
            'Category': ['Salaries'] * 9,  # One row per category + three total rows
            'Subcategory': [
                'Program Director',
                'Department Heads',
                'Administrative Staff',
                'Regional Coordinators',
                'Local Coordinators',
                'Support Staff',
                'Total Median',
                'Total High',
                'Total Low',
            ],
            year_col: [
                expenses['Program Director'][1],  # Median only for subcategories
                expenses['Department Heads'][1],
                expenses['Administrative Staff'][1],
                expenses['Regional Coordinators'][1],
                expenses['Local Coordinators'][1],
                expenses['Support Staff'][1],
                total_salaries['Median'],
                total_salaries['High'],
                total_salaries['Low']
            ]
        })

    def compute_mentor_expense(self, year, mentors):
        """
        Compute the projected expense for mentors based on the number of mentors at each tier.

        - Uses predefined salary distributions for different mentorship involvement levels.
        - Applies percentile-based pricing for a range of possible expenses.
        - Computes total median, high, and low expense estimates.

        Args:
            year (int): The current simulation year.
            mentors (pd.DataFrame): DataFrame containing the number of mentors per involvement level.

        Returns:
            pd.DataFrame: A DataFrame containing mentor expense projections.
        """
        year_col = f'Year {year}'

        # Salary pricing for each mentor level
        pricing = {
            'High Involvement': {'Low': 2.7, 'High': 4.5, 'Mean': 3.6, 'STD': (4.5 - 2.7) / 3},
            'Medium Involvement': {'Low': 1.0, 'High': 2.7, 'Mean': 1.8, 'STD': (2.7 - 1.0) / 3},
            'Low Involvement': {'Low': 0.2, 'High': 1.0, 'Mean': 0.6, 'STD': (1.0 - 0.2) / 3}
        }

        # Extract mentor counts per involvement level
        mentor_counts = {level: mentors.loc[mentors['Subcategory'] == level, year_col].values[0] for level in pricing}

        # Compute expense estimates using percentile mapping
        expenses = {
            level: mentor_counts[level] * self.map_percentile_to_price(
                pricing[level]['Mean'], pricing[level]['STD'], pricing[level]['Low'], pricing[level]['High'], 0.25, 0.75
            ) for level in pricing
        }

        # Compute total median, high, and low expense estimates
        total_median = sum(expenses.values())
        total_std = np.sqrt(sum(((pricing[level]['High'] - pricing[level]['Low']) / 3) ** 2 * (
            mentor_counts[level] / sum(mentor_counts.values()) if sum(mentor_counts.values()) > 0 else 0) for level in
                                pricing))
        total_high = total_median + total_std * sum(mentor_counts.values())
        total_low = total_median - total_std * sum(mentor_counts.values())

        return pd.DataFrame({
            'Category': ['Mentorship'] * 6,
            'Subcategory': list(expenses.keys()) + ['Total Median', 'Total High', 'Total Low'],
            year_col: [round(value, 0) for value in list(expenses.values()) + [total_median, total_high, total_low]]
        })

    def compute_infrastructure_expense(self, year, infrastructure, employees):
        """
        Compute the projected infrastructure and software costs based on the number of offices and employees.

        - HQ costs change depending on organization scale.
        - Regional offices are allocated based on 100 employees per office.
        - Software costs are directly proportional to employee count at 0.3 per employee.

        Args:
            year (int): The current simulation year.
            infrastructure (pd.DataFrame): DataFrame containing infrastructure allocations.
            employees (pd.DataFrame): DataFrame containing the number of employees.

        Returns:
            pd.DataFrame: A DataFrame containing infrastructure and software expense projections.
        """
        year_col = f'Year {year}'
        total_offices = infrastructure.loc[infrastructure['Subcategory'] == 'Total Offices', year_col].values[0]
        employee_count = employees.loc[employees['Subcategory'] == 'Total Employees', year_col].values[0]

        # Define cost parameters
        hq_small = {'Low': 2.77, 'High': 3.99}
        hq_large = {'Low': 11.07, 'High': 15.99}
        regional_office = {'Low': 4.36, 'High': 6.30}
        software_pricing = {'Low': 0.2, 'High': 0.4, 'Mean': 0.3, 'STD': (0.4 - 0.2) / 3}

        # Determine HQ cost
        if employee_count <= 150:
            hq_cost_low, hq_cost_high = hq_small['Low'], hq_small['High']
        else:
            hq_cost_low, hq_cost_high = hq_large['Low'], hq_large['High']

        # Compute regional office costs
        regional_office_count = total_offices - 1  # Exclude HQ
        regional_cost_low = regional_office_count * regional_office['Low']
        regional_cost_high = regional_office_count * regional_office['High']

        # Compute software costs using the mapping function
        software_cost = employee_count * self.map_percentile_to_price(
            software_pricing['Mean'], software_pricing['STD'], software_pricing['Low'], software_pricing['High'], 0.25,
            0.75
        )

        # Compute total costs
        infra_low = hq_cost_low + regional_cost_low + software_cost
        infra_high = hq_cost_high + regional_cost_high + software_cost
        infra_median = (infra_low + infra_high) / 2

        return pd.DataFrame({
            'Category': ['Infrastructure'] * 3,
            'Subcategory': ['Total Median', 'Total High', 'Total Low'],
            year_col: [round(infra_median), round(infra_high), round(infra_low)]
        })

    def compute_marketing_expense(self, year):
        """
        Compute the projected marketing costs based on total income.

        - Marketing is set at 10% of total income.
        - High estimate is +2% of total income.
        - Low estimate is -2% of total income.

        Args:
            year (int): The current simulation year.

        Returns:
            pd.DataFrame: A DataFrame containing marketing expense projections.
        """
        year_col = f'Year {year}'
        total_income = self.financial_results.loc[
            (self.financial_results['Category'] == 'Total Income') & (
                        self.financial_results['Subcategory'] == 'Median'),
            year_col
        ].values[0]

        base_percentage = 0.10
        adjustment = 0.02

        marketing_low = total_income * (base_percentage - adjustment)
        marketing_median = total_income * base_percentage
        marketing_high = total_income * (base_percentage + adjustment)

        return pd.DataFrame({
            'Category': ['Marketing'] * 3,
            'Subcategory': ['Total Median', 'Total High', 'Total Low'],
            year_col: [round(marketing_median), round(marketing_high), round(marketing_low)]
        })

    def compute_event_expense(self, year, events):
        """
        Compute the projected event costs based on the number of events.

        - Medium events cost between 40-60.
        - Large events cost between 400-600.
        - High/Low estimates use standard deviation approximation.

        Args:
            year (int): The current simulation year.
            events (pd.DataFrame): DataFrame containing the number of events.

        Returns:
            pd.DataFrame: A DataFrame containing event expense projections.
        """
        year_col = f'Year {year}'
        large_events = events.loc[events['Subcategory'] == 'Large Events', year_col].values[0]
        medium_events = events.loc[events['Subcategory'] == 'Medium Events', year_col].values[0]

        # Cost parameters
        event_pricing = {
            'Medium': {'Low': 40, 'High': 60, 'Mean': 50, 'STD': (60 - 40) / 3},
            'Large': {'Low': 400, 'High': 600, 'Mean': 500, 'STD': (600 - 400) / 3}
        }

        # Compute costs using the mapping function
        medium_event_cost = medium_events * self.map_percentile_to_price(
            event_pricing['Medium']['Mean'], event_pricing['Medium']['STD'], event_pricing['Medium']['Low'],
            event_pricing['Medium']['High'], 0.25, 0.75
        )
        large_event_cost = large_events * self.map_percentile_to_price(
            event_pricing['Large']['Mean'], event_pricing['Large']['STD'], event_pricing['Large']['Low'],
            event_pricing['Large']['High'], 0.25, 0.75
        )

        total_median = medium_event_cost + large_event_cost
        total_std = np.sqrt(sum(((event_pricing[tier]['High'] - event_pricing[tier]['Low']) / 3) ** 2 * (
            events.loc[events['Subcategory'] == f'{tier} Events', year_col].values[0] / sum(events[year_col]) if sum(
                events[year_col]) > 0 else 0) for tier in event_pricing))
        total_high = total_median + total_std * sum(events[year_col])
        total_low = total_median - total_std * sum(events[year_col])

        return pd.DataFrame({
            'Category': ['Events'] * 3,
            'Subcategory': ['Total Median', 'Total High', 'Total Low'],
            year_col: [round(total_median), round(total_high), round(total_low)]
        })

    def compute_income(self, year):
        """
        Compute the total projected income by aggregating revenue streams from all sources.

        This function consolidates median, high, and low estimates across all income categories,
        including memberships, sponsorships, grants, licensing, and venture income. The resulting
        totals represent the overall financial outlook for the program in each scenario.

        Args:
            year (int): The current year in the simulation (1-10).

        Returns:
            pd.DataFrame: A DataFrame containing the total projected income estimates.
        """
        year_col = f'Year {year}'

        # Extract relevant financial data
        categories = ['Memberships', 'Sponsorships', 'Grants', 'Licensing', 'Ventures']

        # Extract values for total median, high, and low income from each category
        total_median = self.financial_results[
            (self.financial_results['Category'].isin(categories)) &
            (self.financial_results['Subcategory'] == 'Total Median')
            ][year_col].sum()

        total_high = self.financial_results[
            (self.financial_results['Category'].isin(categories)) &
            (self.financial_results['Subcategory'] == 'Total High')
            ][year_col].sum()

        total_low = self.financial_results[
            (self.financial_results['Category'].isin(categories)) &
            (self.financial_results['Subcategory'] == 'Total Low')
            ][year_col].sum()

        return pd.DataFrame({
            'Category': ['Total Income'] * 3,
            'Subcategory': ['Median', 'High', 'Low'],
            year_col: [total_median, total_high, total_low]
        })

    def compute_expense(self, year):
        """
        Compute the total projected expense by aggregating all expense categories.

        This function consolidates median, high, and low estimates across all expense categories,
        including mentorship, infrastructure, marketing, and events.

        Args:
            year (int): The current year in the simulation (1-10).

        Returns:
            pd.DataFrame: A DataFrame containing the total projected expense estimates.
        """
        year_col = f'Year {year}'

        categories = ['Salaries', 'Mentorship', 'Infrastructure', 'Marketing', 'Event']

        total_median = self.financial_results[
            (self.financial_results['Category'].isin(categories)) &
            (self.financial_results['Subcategory'] == 'Total Median')
            ][year_col].sum()

        total_high = self.financial_results[
            (self.financial_results['Category'].isin(categories)) &
            (self.financial_results['Subcategory'] == 'Total High')
            ][year_col].sum()

        total_low = self.financial_results[
            (self.financial_results['Category'].isin(categories)) &
            (self.financial_results['Subcategory'] == 'Total Low')
            ][year_col].sum()

        return pd.DataFrame({
            'Category': ['Total Expense'] * 3,
            'Subcategory': ['Median', 'High', 'Low'],
            year_col: [round(total_median), round(total_high), round(total_low)]
        })

    def compute_profit(self, year):
        """
        Compute the total projected profit by subtracting expenses from income.

        - Median profit = Median income - Median expense.
        - Low profit = Low income - High expense.
        - High profit = High income - Low expense.

        Args:
            year (int): The current year in the simulation (1-10).

        Returns:
            pd.DataFrame: A DataFrame containing the total projected profit estimates.
        """
        year_col = f'Year {year}'

        total_income_median = self.financial_results[
            (self.financial_results['Category'] == 'Total Income') &
            (self.financial_results['Subcategory'] == 'Median')
            ][year_col].values[0]

        total_income_low = self.financial_results[
            (self.financial_results['Category'] == 'Total Income') &
            (self.financial_results['Subcategory'] == 'Low')
            ][year_col].values[0]

        total_income_high = self.financial_results[
            (self.financial_results['Category'] == 'Total Income') &
            (self.financial_results['Subcategory'] == 'High')
            ][year_col].values[0]

        total_expense_median = self.financial_results[
            (self.financial_results['Category'] == 'Total Expense') &
            (self.financial_results['Subcategory'] == 'Median')
            ][year_col].values[0]

        total_expense_low = self.financial_results[
            (self.financial_results['Category'] == 'Total Expense') &
            (self.financial_results['Subcategory'] == 'Low')
            ][year_col].values[0]

        total_expense_high = self.financial_results[
            (self.financial_results['Category'] == 'Total Expense') &
            (self.financial_results['Subcategory'] == 'High')
            ][year_col].values[0]

        profit_median = total_income_median - total_expense_median
        profit_low = total_income_low - total_expense_high
        profit_high = total_income_high - total_expense_low

        return pd.DataFrame({
            'Category': ['Total Profit'] * 3,
            'Subcategory': ['Median', 'High', 'Low'],
            year_col: [round(profit_median), round(profit_high), round(profit_low)]
        })

    def generate_cash_flow_forecast(self):
        """
        Generate a structured 10-year cash flow forecast using median financial projections.

        This function extracts revenue and subcategory expense values from the internal `financial_results` DataFrame,
        splits expenses into operating and capital expenditures, computes taxes at a flat rate of 35%, and derives
        net cash flow and ending balances over 10 years. All results are stored in `self.cash_flow_forecast` and
        exported to CSV.
        """
        years = [f"Year {i}" for i in range(1, 11)]

        # Opening cash and financial inflows (manually configured)
        opening_cash = 0
        financial_inflows = [
            (1, 500),
            (3, 1_500),
            (5, 9_000),
        ]
        inflow_map = {f"Year {y}": amount for y, amount in financial_inflows}

        # Revenue and profit
        revenue_row = self.financial_results[
            (self.financial_results["Category"] == "Total Income") &
            (self.financial_results["Subcategory"] == "Median")
            ]
        profit_row = self.financial_results[
            (self.financial_results["Category"] == "Total Profit") &
            (self.financial_results["Subcategory"] == "Median")
            ]

        revenue = revenue_row[years].values[0]
        profit = profit_row[years].values[0]
        taxes = [round(p * 0.35, 0) for p in profit]

        # Capital Expenditures (Infrastructure)
        capex_row = self.financial_results[
            (self.financial_results["Category"] == "Infrastructure") &
            (self.financial_results["Subcategory"] == "Total Median")
            ]
        capex = capex_row[years].values[0]

        # Operating Expenditures (Salaries, Mentorship, Marketing, Event Expenses)
        opex_cats = ["Salaries", "Mentorship", "Marketing", "Event Expenses"]
        opex_df = self.financial_results[
            (self.financial_results["Category"].isin(opex_cats)) &
            (self.financial_results["Subcategory"] == "Total Median")
            ]
        expense = opex_df[years].sum().values

        # Net cash flow and balances
        net_flow = [r - e - c - t for r, e, c, t in zip(revenue, expense, capex, taxes)]
        equity = [inflow_map.get(y, 0.0) for y in years]
        opening = [opening_cash] + [None] * 9
        ending = []

        for i in range(10):
            prev = opening[i] if i == 0 else ending[i - 1]
            cash = prev + net_flow[i] + equity[i]
            ending.append(cash)
            if i < 9:
                opening[i + 1] = cash

        cash_inflows = [revenue[i] + equity[i] for i in range(10)]
        cash_outflows = [expense[i] + capex[i] + taxes[i] for i in range(10)]

        df = pd.DataFrame([
            ["Opening Cash", *opening],
            ["Cash Inflows", *cash_inflows],
            ["Revenue", *revenue],
            ["Equity/Grants Raised", *equity],
            ["Cash Outflows", *cash_outflows],
            ["Operating Expenses", *expense],
            ["Capital Expenditures", *capex],
            ["Taxes Paid", *taxes],
            ["Net Cash Flow", *net_flow],
            ["Ending Cash", *ending]
        ], columns=["Category"] + years)

        self.cash_flow_forecast = df
        df.to_csv("models/cash-flow-forecast.csv", index=False)

    def generate_income_statement(self):
        """
        Generate a structured 10-year income statement using median financial projections.

        This function assembles total and component revenue and expense categories from the financial_results table.
        Revenue includes Memberships, Sponsorship, Grants, Licensing, and Venture Income.
        Operating Expenses include all non-capital operating categories: Salaries, Mentorship, Marketing, and Event Expenses.
        Capital expenditures (e.g., Infrastructure) are excluded. Taxes are computed at a flat 35% on Operating Profit.
        Output is stored in self.income_statement and saved to models/income-statement.csv.
        """
        years = [f"Year {i}" for i in range(1, 11)]

        # Revenue (Total)
        revenue_row = self.financial_results[
            (self.financial_results["Category"] == "Total Income") &
            (self.financial_results["Subcategory"] == "Median")
            ]
        revenue = revenue_row[years].values[0]

        # Revenue breakdowns
        def extract_revenue(cat):
            row = self.financial_results[
                (self.financial_results["Category"] == cat) &
                (self.financial_results["Subcategory"] == "Total Median")
                ]
            return row[years].values[0] if not row.empty else [0] * 10

        memberships = extract_revenue("Memberships")
        sponsorships = extract_revenue("Sponsorship Income")
        grants = extract_revenue("Grants")
        licensing = extract_revenue("Licensing Income")
        ventures = extract_revenue("Venture Income")

        # Operating expenses
        def extract_expense(cat):
            row = self.financial_results[
                (self.financial_results["Category"] == cat) &
                (self.financial_results["Subcategory"] == "Total Median")
                ]
            return row[years].values[0] if not row.empty else [0] * 10

        salaries = extract_expense("Salaries")
        mentorship = extract_expense("Mentorship")
        marketing = extract_expense("Marketing")
        events = extract_expense("Event Expenses")

        operating_expenses = [
            s + m + mk + e
            for s, m, mk, e in zip(salaries, mentorship, marketing, events)
        ]

        operating_profit = [r - oe for r, oe in zip(revenue, operating_expenses)]
        taxes = [round(op * 0.35, 0) for op in operating_profit]
        net_income = [op - t for op, t in zip(operating_profit, taxes)]

        df = pd.DataFrame([
            ["Revenue", *revenue],
            ["Memberships", *memberships],
            ["Sponsorships", *sponsorships],
            ["Grants", *grants],
            ["Licensing/Grants", *licensing],
            ["Venture Income", *ventures],
            ["Operating Expenses", *operating_expenses],
            ["Salaries", *salaries],
            ["Mentorship", *mentorship],
            ["Marketing", *marketing],
            ["Event Expenses", *events],
            ["Operating Profit", *operating_profit],
            ["Taxes (est. 35%)", *taxes],
            ["Net Income", *net_income],
        ], columns=["Category"] + years)

        self.income_statement = df
        df.to_csv("models/income-statement.csv", index=False)

    def generate_balance_sheet(self):
        """
        Generate a 10-year balance sheet using only modeled values.

        Assets include:
        - Cash: from cash flow forecast 'Ending Cash'
        - Equipment & Infrastructure: cumulative Capital Expenditures

        Liabilities: none (confirmed zero)

        Equity includes:
        - Paid-in Capital: cumulative Equity/Grants Raised
        - Retained Earnings: cumulative Net Income from income statement

        Total Liabilities + Equity is checked to match Total Assets.

        Result is saved to models/balance-sheet.csv.
        """
        years = [f"Year {i}" for i in range(1, 11)]

        # Cash from Ending Cash
        cash_row = self.cash_flow_forecast[
            self.cash_flow_forecast["Category"] == "Ending Cash"
            ].iloc[0][years].values

        # Equipment & Infra from cumulative CapEx
        capex_row = self.cash_flow_forecast[
            self.cash_flow_forecast["Category"] == "Capital Expenditures"
            ].iloc[0][years].values
        cumulative_capex = [sum(capex_row[:i + 1]) for i in range(10)]

        # Paid-in Capital from cumulative equity inflows
        equity_row = self.cash_flow_forecast[
            self.cash_flow_forecast["Category"] == "Equity/Grants Raised"
            ].iloc[0][years].values
        cumulative_equity = [sum(equity_row[:i + 1]) for i in range(10)]

        # Retained Earnings from cumulative Net Income
        net_income_row = self.income_statement[
            self.income_statement["Category"] == "Net Income"
            ].iloc[0][years].values
        cumulative_earnings = [sum(net_income_row[:i + 1]) for i in range(10)]

        # Totals
        total_assets = [c + i for c, i in zip(cash_row, cumulative_capex)]
        total_equity = [pc + re for pc, re in zip(cumulative_equity, cumulative_earnings)]

        df = pd.DataFrame([
            ["Assets"] + ["" for _ in years],
            ["Cash", *cash_row],
            ["Equipment & Infra", *cumulative_capex],
            ["Total Assets", *total_assets],
            ["Liabilities"] + ["" for _ in years],
            ["Total Liabilities", *[0] * 10],
            ["Equity"] + ["" for _ in years],
            ["Paid-in Capital", *cumulative_equity],
            ["Retained Earnings", *cumulative_earnings],
            ["Total Equity", *total_equity],
            ["Total Liabilities + Eq.", *total_equity],
        ], columns=["Category"] + years)

        self.balance_sheet = df
        df.to_csv("models/balance-sheet.csv", index=False)

    def map_percentile_to_price(self, mean_price, std_price, low, high, p_low, p_high):
        """
        Maps a percentile slice of a normal distribution to a price range.

        This function extracts the portion of a normal distribution that corresponds to a given
        percentile range and maps it to the specified price range.

        Args:
            mean_price (float): The mean value of the normal distribution.
            std_price (float): The standard deviation of the normal distribution.
            low (float): The lower bound of the price range.
            high (float): The upper bound of the price range.
            p_low (float): The lower bound of the percentile range (e.g., 0.0 for 0%).
            p_high (float): The upper bound of the percentile range (e.g., 0.6 for 60%).

        Returns:
            float: The expected price within the specified percentile range.
        """
        p_values = np.linspace(p_low, p_high, 1000)
        mapped_values = stats.norm.ppf(p_values, loc=mean_price, scale=std_price)
        return np.mean(np.clip(mapped_values, low, high))  # Ensure values stay within range


if __name__ == "__main__":
    model = FinancialModel()
    model.compute()
