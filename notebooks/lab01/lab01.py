import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import polars as pl
    import altair as alt
    import numpy as np

    return alt, np, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Lab 01: Campaign Finance & Ideology Estimation with PCA

    In this lab you will explore campaign finance data from the 2024 election cycle
    and use **Principal Component Analysis (PCA)** to estimate the political ideology
    of donors and recipients.

    The data comes from the [Database on Ideology, Money in Politics, and Elections
    (DIME)](https://data.stanford.edu/dime), compiled by Adam Bonica. We are working
    with a subset that includes donors who contributed to **8 or more** distinct
    candidates or committees.

    **What you'll do:**

    1. **Part A** — Explore donation patterns: who gives, who receives, and how
       contributions break down by party and industry.
    2. **Part B** — Estimate ideology scores using PCA on a donor × recipient
       contribution matrix, following a simplified version of Bonica's
       approach.

    **Key columns in the data:**

    | Column | Description |
    |--------|-------------|
    | `bonica_cid` | Unique donor ID |
    | `bonica_rid` | Unique recipient ID |
    | `total_amount` | Total dollars from this donor to this recipient |
    | `contributor_type` | `I` = individual, `C` = committee |
    | `recipient_party` | `100` = Democrat, `200` = Republican |
    | `recipient_type` | `CAND` = candidate, `COMM` = committee |
    | `seat` | Office sought (e.g. `federal:house`, `federal:senate`) |
    | `contributor_occupation` | Self-reported occupation |
    """)
    return


@app.cell(hide_code=True)
def _(mo, pl):
    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    _nb_dir = mo.notebook_location()
    _data_dir = _nb_dir / "public" / "data"

    contributions = pl.read_parquet(
        str(_data_dir / "contributions_merged.parquet")
    ).filter(
        # Keep only individual donors (remove PACs and committees)
        (pl.col("contributor_type") == "I")
        # Remove fundraising conduits (not substantively interesting)
        & ~pl.col("recipient_name").str.to_lowercase().str.contains("actblue")
        & ~pl.col("recipient_name").str.to_lowercase().str.contains("winred")
    )
    matrix_df = pl.read_parquet(str(_data_dir / "contribution_matrix.parquet"))
    recipients_meta = pl.read_parquet(str(_data_dir / "recipients.parquet"))
    contributors_meta = pl.read_parquet(str(_data_dir / "contributors.parquet"))
    occ_industry = pl.read_parquet(str(_data_dir / "occupation_industry.parquet"))

    mo.vstack(
        [
            mo.md(
                f"""
    **Data loaded successfully!**

    | Dataset | Rows | Columns |
    |---------|------|---------|
    | Raw contributions | {contributions.shape[0]:,} | {contributions.shape[1]} |
    | Contribution matrix | {matrix_df.shape[0]:,} | {matrix_df.shape[1]} |
    | Recipients metadata | {recipients_meta.shape[0]:,} | {recipients_meta.shape[1]} |
    | Contributors metadata | {contributors_meta.shape[0]:,} | {contributors_meta.shape[1]} |
    | Occupation to Industry | {occ_industry.shape[0]:,} | {occ_industry.shape[1]} |
    """
            ),
            mo.accordion(
                {
                    "Preview: recipients_meta": mo.plain_text(
                        str(recipients_meta.head(5))
                    ),
                    "Preview: occ_industry": mo.plain_text(
                        str(occ_industry.head(5))
                    ),
                }
            ),
        ]
    )
    return (
        contributions,
        contributors_meta,
        matrix_df,
        occ_industry,
        recipients_meta,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## How to Complete Exercises

    Each exercise contains placeholder code marked with `...` that you need to replace with your solution. The exercises use `mo.stop()` to pause execution until you complete each step.

    **To complete an exercise:**

    1. Replace the `...` placeholder with your code
    2. Run the cell — the `mo.stop()` checks automatically pass once `...` is replaced with a real value, so you don't need to delete anything

    For example, if you see:

    ```python
    my_result = ...  # YOUR CODE

    mo.stop(
        my_result is ...,
        mo.md("⚠️ Complete this step...")
    )
    ```

    Just replace the `...` with your code:

    ```python
    my_result = some_data.filter(pl.col("x") > 0)  # Your actual code

    mo.stop(
        my_result is ...,       # ← this check now passes automatically
        mo.md("⚠️ Complete this step...")
    )
    ```

    > **Tip:** Work through each step sequentially. The `mo.stop()` pattern ensures you get feedback at each stage before moving on.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    ## Part A: Exploring Campaign Finance Data

    Before jumping into PCA, let's build intuition about who gives money in
    American politics, who receives it, and how those flows relate to party.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Example 1: Top 20 donors by total amount

    Each row in `contributions` is a donor-recipient pair with the total dollars
    flowing from that donor to that recipient. To find the biggest donors we
    group by `bonica_cid`, sum the amounts, sort, and take the top 20.
    """)
    return


@app.cell
def _(alt, contributions, pl):
    # Example 1: Top 20 donors by total amount donated
    top_donors = (
        contributions.group_by("bonica_cid", "contributor_name")
        .agg(pl.col("total_amount").sum().alias("total_donated"))
        .sort("total_donated", descending=True)
        .head(20)
    )

    alt.Chart(
        top_donors, title="Top 20 Donors by Total Contributions"
    ).mark_bar().encode(
        x=alt.X("total_donated:Q", title="Total Donated ($)"),
        y=alt.Y("contributor_name:N", sort="-x", title=None),
    ).properties(width=600, height=400)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Exercise 0: MIT donors

    **Task:** Filter `contributions` to donors employed at MIT and find the
    **top 5 donors** by total amount.

    *Steps:*

    1. Filter `contributions` to rows where `contributor_employer` contains
       `"mit"` (use `.str.to_lowercase().str.contains("mit")`).
    2. Group by `bonica_cid` and `contributor_name`, summing `total_amount`.
    3. Sort descending and take the top 5.

    **Question:** Who is MIT's biggest political donor?
    """)
    return


@app.cell
def _(contributions, mo, pl):
    # ---- Exercise 0 ----
    mit_top_donors = (
        contributions
        .filter(pl.col("contributor_employer").str.to_lowercase().str.contains("mit"))
        .group_by("bonica_cid", "contributor_name")
        .agg(pl.col("total_amount").sum())
        .sort("total_amount", descending=True)
        .head(5)
    )


    mo.stop(
        mit_top_donors is ...,
        mo.md(
            "⚠️ **Complete Exercise 0:** Filter to MIT employees and find the top 5 donors."
        ),
    )

    mit_top_donors
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Reflection:** Who is MIT's biggest political donor? Were you surprised?

    MIT's biggest political donor is Ronald Rivest, and I was suprised to find out a cryptographer donated the most money, and by much more than the next political donor.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Exercise 1: In-state vs. out-of-state donations

    **Task:** For House and Senate candidates, compute the share of
    donations that come from **within the candidate's state** vs. from
    **out of state**. Break down by party.

    *Steps:*

    1. Filter to House and Senate candidates
       (`seat.is_in(["federal:house", "federal:senate"])`) and the two major
       parties.
    2. Add an `in_state` column that is `True` when `contributor_state`
       equals `recipient_state`.
    3. Group by `recipient_party` and `in_state`, summing `total_amount`.
    4. Add readable labels for party and in-state status.
    5. Create a **normalized stacked bar chart** (use `stack="normalize"`
       in the x encoding) showing the in-state share for each party.

    **Question:** Which party receives a larger share of its money from
    out-of-state donors? What does that tell you about the
    "nationalization" of campaign finance?
    """)
    return


@app.cell
def _(alt, contributions, mo, pl):
    # ---- Exercise 1 ----
    # Steps 1-2: Filter to House/Senate candidates and the two major parties,
    # add an in_state column, group by party and in_state, and add labels.
    _congressional = (
        contributions
        .filter(
            pl.col("seat").is_in(["federal:house","federal:senate"]),
            pl.col("recipient_party").is_in(["100","200"])
        )
    )  # YOUR CODE: filter contributions on seat and recipient_party

    # Add an in_state column (True when contributor_state == recipient_state)
    # then group by recipient_party and in_state, summing total_amount.
    # Add readable labels for party and in_state.
    in_out_state = (
        _congressional
        .with_columns(
            pl.when(pl.col("contributor_state") == pl.col("recipient_state"))
                .then(pl.lit("In state"))
                .otherwise(pl.lit("Out of state"))
            .alias("in_state"),
            pl.col("recipient_party")
                .replace({100: "Democrat", 200: "Republican"})
        )
        .group_by("recipient_party", "in_state")
        .agg(pl.col("total_amount").sum())
    )  # YOUR CODE

    mo.stop(
        _congressional is ... or in_out_state is ...,
        mo.md(
            "⚠️ **Complete Exercise 1, Steps 1-2:** Filter `contributions` to House/Senate candidates and the two major parties, then create `in_out_state` by adding an `in_state` column, grouping, and adding labels."
        ),
    )

    # Step 3: Create a normalized stacked bar chart
    _ex1_chart = alt.Chart(in_out_state).mark_bar().encode(
        x=alt.X('total_amount:Q', stack="normalize", title="Share of Total Amount"),
        y = alt.Y("recipient_party:N", title="Recipient Party"),
        color=alt.Color("in_state:N", title="Location")
    )  # YOUR CODE

    mo.stop(
        _ex1_chart is ...,
        mo.md(
            "⚠️ **Complete Exercise 1, Step 3:** Create a normalized stacked bar chart."
        ),
    )

    _ex1_chart
    return


@app.cell
def _(mo):
    mo.md("""
    **Reflection:** Which party receives a larger share from out-of-state donors? What does that tell you about the nationalization of campaign finance?

    Republicans receive a larger share from out-of-state donors, though it is off by a small margin from Democrats. This shows that campaign financing has largely gone national with state candidates receiving money from all over the country by people that want to fund one party's interests. Rather than each candidate strictly represent their particular state, since donors want national political power through a majority, candidates represent the interests of, engaged high-income people across the nation.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Example 2: Donation split by party for top occupations

    This example shows how different occupations split their donations between
    the two major parties. We compute the share going to each party for the
    10 most common occupations.
    """)
    return


@app.cell
def _(alt, contributions, pl):
    # Example 2: Top occupations and their party split
    _party_occ = (
        contributions.filter(
            pl.col("recipient_party").is_in(["100", "200"])
            & pl.col("contributor_occupation").is_not_null()
        )
        .group_by(["contributor_occupation", "recipient_party"])
        .agg(pl.col("total_amount").sum().alias("total"))
    )

    # Find top 10 occupations by total donations
    _top_occs = (
        _party_occ.group_by("contributor_occupation")
        .agg(pl.col("total").sum().alias("grand_total"))
        .sort("grand_total", descending=True)
        .head(10)
        .select("contributor_occupation")
    )

    occ_party_split = _party_occ.join(
        _top_occs, on="contributor_occupation"
    ).with_columns(
        pl.when(pl.col("recipient_party") == "100")
        .then(pl.lit("Democrat"))
        .otherwise(pl.lit("Republican"))
        .alias("party")
    )

    alt.Chart(
        occ_party_split, title="Donation Split by Party - Top 10 Occupations"
    ).mark_bar().encode(
        x=alt.X("total:Q", title="Total Donated ($)", stack="normalize"),
        y=alt.Y("contributor_occupation:N", sort="-x", title=None),
        color=alt.Color(
            "party:N",
            scale=alt.Scale(
                domain=["Democrat", "Republican"], range=["#2166ac", "#b2182b"]
            ),
            title="Party",
        ),
    ).properties(width=600, height=350)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Exercise 2: Industry donation patterns

    **Task:** Use the `occ_industry` mapping to compute what share of each
    industry's donations go to Democrats vs. Republicans.

    *Steps:*

    1. Join `contributions` with `occ_industry` on `contributor_occupation` (use `how="left"`).
    2. Filter to the two major parties and non-null industries.
    3. Group by `industry` and `recipient_party`, summing `total_amount`.
    4. Add a readable `party` label (like in Example 2).
    5. Create a normalized stacked bar chart (use `stack="normalize"` in the x encoding).

    **Question:** Which industries lean most Democratic? Most Republican?
    """)
    return


@app.cell
def _(alt, contributions, mo, occ_industry, pl):
    # ---- Exercise 2 ----

    # Step 1: Join contributions with the occupation-to-industry mapping
    _with_industry = contributions.join(
        occ_industry, on="contributor_occupation", how="left"
    )

    # Step 2: Filter to major parties and non-null industries
    _filtered = (
        _with_industry
        .filter(
            pl.col("recipient_party").is_in(["100", "200"])
            & pl.col("industry").is_not_null()
        )
    )  # YOUR CODE: filter _with_industry with two conditions

    mo.stop(
        _filtered is ...,
        mo.md(
            "⚠️ **Complete Exercise 2, Step 2:** Filter `_with_industry` to major parties and non-null industries."
        ),
    )

    # Step 3: Group by industry and party, sum amounts, add party label
    industry_party = (
        _filtered
        .group_by("industry", "recipient_party")
        .agg(pl.col("total_amount").sum())
        .with_columns(
            pl.when(pl.col("recipient_party") == "100")
            .then(pl.lit("Democrat"))
            .otherwise(pl.lit("Republican"))
            .alias("party")
        )
    ) # YOUR CODE

    mo.stop(
        industry_party is ...,
        mo.md(
            "⚠️ **Complete Exercise 2, Step 3:** Create `industry_party` by grouping and adding a party label."
        ),
    )

    # Step 4: Create a normalized stacked bar chart
    _ex2_chart = alt.Chart(
        industry_party, title="Donation Split by Party - Industries"
    ).mark_bar().encode(
        x=alt.X("total_amount:Q", title="Total Donated ($)", stack="normalize"),
        y=alt.Y("industry:N", sort="-x", title=None),
        color=alt.Color(
            "party:N",
            scale=alt.Scale(
                domain=["Democrat", "Republican"], range=["#2166ac", "#b2182b"]
            ),
            title="Party",
        ),
    ).properties(width=600, height=350)  # YOUR CODE

    mo.stop(
        _ex2_chart is ...,
        mo.md(
            "⚠️ **Complete Exercise 2, Step 4:** Create a normalized stacked bar chart."
        ),
    )

    _ex2_chart
    return


@app.cell
def _(mo):
    mo.md("""
    **Reflection:** Which industries lean most Democratic? Most Republican? What might explain these patterns?

    Arts, academic, and technological industries lean more Democratic (with unemployed leaning most Democratic), while business and finance industries lean more Republican. One explanation might be that Republicans are more opposed to business regulation than Democrats are, which can explain why business and management industries might want to support Republicans over Democrats. On the other hand, Democrats are more supportive of welfare programs and academic advancement/accessibility, which can explain why the unemployed along with research/educational industries might support them more.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Exercise 3: Open-ended exploration

    **Pick one** of these questions and write Polars code + an Altair chart
    to answer it:

    **(a)** How do employees of well-known companies donate? Pick 5-8
    companies (e.g. Google, Walmart, Exxon, Amazon, Goldman Sachs) and
    compare their party split. Filter using
    `pl.col("contributor_employer").str.to_lowercase().str.contains(...)`.

    **(b)** How does average donation size vary across the top 10 occupations?

    **(c)** Investigate a question of your own choosing.
    """)
    return


@app.cell
def _(alt, contributions, mo, pl):
    # ---- Exercise 3 ----
    # Pick one of the questions from the prompt and implement your analysis here.

    # (b) 
    # using code from above
    _party_occ = (
        contributions.filter(
            pl.col("recipient_party").is_in(["100", "200"])
            & pl.col("contributor_occupation").is_not_null()
        )
        .group_by(["contributor_occupation", "recipient_party"])
        .agg(pl.col("total_amount").sum().alias("total"))
    )

    # Find top 10 occupations by total donations
    _top_occs = (
        _party_occ.group_by("contributor_occupation")
        .agg(pl.col("total").sum().alias("grand_total"))
        .sort("grand_total", descending=True)
        .head(10)
        .select("contributor_occupation")
    )

    _party_occ_avg = (
        contributions.filter(
            pl.col("contributor_occupation").is_in(_top_occs["contributor_occupation"]),
            pl.col("recipient_party").is_in(["100", "200"])
        )
        .group_by(["contributor_occupation", "recipient_party"])
        .agg(pl.col("total_amount").mean().alias("avg_donation"))
        .with_columns(
            pl.col("recipient_party").replace({100: "Democrat", 200: "Republican"}).alias("party")
        )
    )
    _ex3_result = alt.Chart(
        _party_occ_avg, title="Average Donation Size by Party - Top 10 Occupations"
    ).mark_bar().encode(
        x=alt.X("avg_donation:Q", title="Average Donated ($)"),
        y=alt.Y("contributor_occupation:N", sort="-x", title=None),
        color=alt.Color(
            "party:N",
            scale=alt.Scale(
                domain=["Democrat", "Republican"], range=["#2166ac", "#b2182b"]
            ),
            title="Party",
        ),
    ).properties(width=600, height=350) # YOUR CODE: analysis + visualization

    mo.stop(
        _ex3_result is ...,
        mo.md(
            "⚠️ **Complete Exercise 3:** Choose one of the open-ended questions and implement your analysis."
        ),
    )

    _ex3_result
    return


@app.cell
def _(mo):
    mo.md("""
    **Reflection:** Summarize your findings. What patterns did you find and what might explain them?

    I found that for the top 10 occupations, Republicans on average donate a larger amount of money to candidates than Democrats. This can be explained by Republicans tending to hold more entrepreneurial (chairman, executive, etc.) and high-income occupations, leading to more money to donate. I also found it really interesting that the 'candidate' occupation completely dominates any other occupation with an average of 5 million donated to nearly strictly Republicans. This is a crazy imbalance that shows that candidates (most likely Republican candidates) donate to other candidates in order to influence and support their decisions and national positions.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Part B: Estimating Ideology with PCA

    We now turn to the main analytical goal: **recovering an ideological
    dimension** from campaign-finance data.

    ### Intuition

    Donors who give to the same candidates probably share similar political
    views. The **contributor x candidate matrix** encodes this information:
    each row is a donor, each column is a candidate, and each cell records
    whether that donor contributed to that candidate.

    We use **binary** (gave / didn't give) rather than dollar amounts. This
    focuses on the *pattern* of who gives to whom rather than how much
    they give - a billionaire and a retiree who support the same set of
    candidates carry the same ideological signal.

    ### Normalizing the matrix

    Even in a binary matrix, some donors give to many candidates (they're
    prolific) and some candidates receive from many donors (they're popular).
    We need to remove this variation so that PCA captures ideology, not
    activity level.

    The idea: compare each cell to what we'd **expect** if donors and
    candidates were matched at random (given their overall activity levels).
    Cells that are higher than expected reveal genuine affinity; cells that
    are lower reveal avoidance.

    Concretely:

    1. Divide every cell by the grand total to get proportions: $P = X / N$
    2. Compute row totals $r$ and column totals $c$ of the proportion matrix.
    3. The **expected** value for cell $(i,j)$ under independence is
       $E_{ij} = r_i \times c_j$.
    4. The **standardized residual** measures how far each cell deviates
       from what we'd expect, scaled by that expectation:

    $$K_{ij} = \frac{P_{ij} - E_{ij}}{\sqrt{r_i \times c_j}}$$

    This removes variation from donor prolificness and candidate popularity,
    leaving the *pattern* of who gives to whom - which is where ideology
    lives. PCA on $K$ should recover the liberal-conservative axis as PC1.
    """)
    return


@app.cell
def _(matrix_df, mo, np):
    # Convert the contribution matrix to a numpy array
    # First column is bonica_cid (donor ID); remaining columns are candidate IDs
    donor_ids = matrix_df["bonica_cid"].to_list()
    recipient_ids = [c for c in matrix_df.columns if c != "bonica_cid"]

    _X_raw = matrix_df.select(recipient_ids).to_numpy().astype(np.float64)

    # Convert to binary: 1 if donor gave to candidate, 0 otherwise
    X = (_X_raw > 0).astype(np.float64)

    mo.md(
        f"""
    ### The contribution matrix

    Our pre-built matrix has **{X.shape[0]:,} donors** (rows) and
    **{X.shape[1]:,} candidates** (columns).

    We convert dollar amounts to **binary** (1 = gave, 0 = didn't give)
    so PCA captures donation *patterns* rather than dollar magnitudes.

    Non-zero fraction: {X.mean():.2%} of cells are 1.
    """
    )
    return X, donor_ids, recipient_ids


@app.cell
def _(mo):
    mo.md(r"""
    ### Exercise 4: Normalizing the matrix

    The code below implements the normalization described above. **Read
    through it carefully** — each step matches a line in the math.

    **Steps (all provided):**

    1. Compute the grand total $N$ (sum of all cells in the matrix).
    2. Compute the proportion matrix $P = X / N$.
    3. Compute row totals $r$ and column totals $c$ of the proportion matrix.
    4. Compute the expected-value matrix $E = r \times c$ (outer product).
    5. Compute the standardized residuals:
       $K_{ij} = (P_{ij} - E_{ij}) / \sqrt{r_i \times c_j}$
    """)
    return


@app.cell
def _(X, mo, np):
    # ---- Exercise 4 ----

    # Step 1: Grand total
    N = X.sum()

    # Step 2: Proportion matrix
    P = X / N

    # Step 3: Row and column totals
    # r has shape (n_donors, 1); c has shape (1, n_recipients)
    r = P.sum(axis=1, keepdims=True)
    c = P.sum(axis=0, keepdims=True)

    # Step 4: Expected values under independence (outer product of totals)
    E = r * c

    # Step 5: Standardized residuals
    # K_ij = (P_ij - E_ij) / sqrt(r_i * c_j)
    K = (P - E) / (np.sqrt(r) * np.sqrt(c))

    # Replace any NaN/Inf from division by zero with 0
    K = np.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)

    mo.md(f"""
    **Normalization complete!**

    - K shape: `{K.shape}`
    - K range: `[{K.min():.4f}, {K.max():.4f}]`
    - K mean: `{K.mean():.6f}`
    """)
    return (K,)


@app.cell
def _(mo):
    mo.md("""
    **Reflection:** In your own words, what does this normalization accomplish? Why do we need to remove the effects of donor prolificness and candidate popularity before running PCA?

    This normalization allows us to recover patterns in the data of donors contributing to candidates, instead of skewing our data with raw amounts of contribution levels varying from donor to donor. Without normalization, donors that choose to donate to lots of candidates they support would dominate another donor that donated to one candidate they support, skewing our data towards prolific donors and losing donors that contribute less. This applies for candidates that receive more votes than others as well. We need to remove these effects because we are trying to capture ideological dimensions, not dimensions relating to raw amounts of support. We would lose the general trend along the ideological dimension if we chose to skew towards donors and candidates that unequally influence the raw distribution of data, catering our PCA analysis towards those with lots of money and/or those that are extremely popular for a period of time.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Exercise 5: Run PCA and interpret the scree plot

    We fit PCA (via SVD) to the standardized residual matrix $K$ with
    `n_components=10`. The code below runs PCA and produces a **scree
    plot** showing how much variance each component explains.

    **Read the scree plot carefully** and answer the reflection question below.
    """)
    return


@app.cell
def _(K, alt, mo, pl):
    from sklearn.decomposition import PCA

    # ---- Exercise 5 ----

    # Fit PCA with 10 components
    pca = PCA(n_components=10)
    scores = pca.fit_transform(K)  # shape (n_donors, 10)

    # Build a DataFrame for the scree plot
    _scree_data = pl.DataFrame(
        {
            "Component": [f"PC{_i + 1}" for _i in range(10)],
            "Variance Explained": pca.explained_variance_ratio_,
        }
    )

    # Scree plot bar chart
    scree_chart = (
        alt.Chart(_scree_data, title="Scree Plot: Variance Explained by Component")
        .mark_bar()
        .encode(
            x=alt.X("Component:N", sort=[f"PC{_i + 1}" for _i in range(10)], title="Component"),
            y=alt.Y("Variance Explained:Q", title="Fraction of Variance Explained"),
        )
        .properties(width=600, height=300)
    )

    mo.vstack(
        [
            mo.md(
                f"**PC1 explains {pca.explained_variance_ratio_[0]:.1%} of the variance.**"
            ),
            scree_chart,
        ]
    )
    return (scores,)


@app.cell
def _(mo):
    mo.md("""
    **Reflection:** Why might PC1 dominate so strongly in campaign finance data? What does the gap between PC1 and PC2 tell us about the structure of American political donations?

    PC1 might dominate strongly in campaign finance data because the fit finds the most direct correlation it can, party affiliation, and represents it as the strongest indicator for who a person might donate to. The gap between PC1 and PC2 tells us that American political donations are guided primarily by which party a person is affiliated with, and any other factors can be seen as inconsistent signals for party donations in comparison.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Exercise 6: Recipient ideology scores

    #### Analytical Goal

    Does PC1 separate Democrats from Republicans? Which end is liberal
    and which is conservative? Build a chart of recipient ideology
    scores to find out.

    #### Background

    We can score each recipient by averaging the PC1 scores of their
    donors. A candidate whose donors are mostly liberal will get a
    score on the liberal end; a candidate whose donors are mostly
    conservative will land on the other side. Candidates who attract
    donors from both parties end up near zero.

    **Tasks:**

    1. Compute each recipient's ideology score (provided below).
    2. Join with `recipients_meta` to get names and parties. Add a readable
       `party` column mapping `"100"` to `"Democrat"` and `"200"` to `"Republican"`.
    3. Take a random sample of ~50 recipients (use `.sample(50)`).
    4. Create a dot chart (use `mark_circle()`) showing these recipients,
       with ideology score on x and recipient name on y, colored by party.
    """)
    return


@app.cell
def _(X, alt, mo, np, pl, recipient_ids, recipients_meta, scores):
    # ---- Exercise 6 ----

    # Step 1 (provided): Compute each recipient's ideology score as the
    # mean PC1 score of their donors.
    # Matrix multiply: (X^T @ donor_scores) gives the sum of donor scores
    # for each recipient; dividing by the donor count gives the mean.
    _donor_pc1 = scores[:, 0]
    _donor_sums = X.T @ _donor_pc1
    _donor_counts = X.sum(axis=0)
    _recipient_scores = _donor_sums / np.maximum(_donor_counts, 1)

    _scores_df = pl.DataFrame(
        {
            "bonica_rid": recipient_ids,
            "ideology_score": _recipient_scores,
        }
    )

    # Step 2: Join with recipients_meta and add a "party" label column
    recipient_ideology = (
        _scores_df.join(recipients_meta, on="bonica_rid")
        .with_columns(
            pl.col("recipient_party").replace({100: "Democrat", 200: "Republican"}).alias("party")
        )
    ) # YOUR CODE

    mo.stop(
        recipient_ideology is ...,
        mo.md(
            "⚠️ **Complete Exercise 6, Step 2:** Join `_scores_df` with `recipients_meta` and add a `party` label column."
        ),
    )

    # Step 3: Take a random sample of 50 recipients
    _sample = recipient_ideology.sample(50)  # YOUR CODE: use .sample(50)

    mo.stop(
        _sample is ...,
        mo.md(
            "⚠️ **Complete Exercise 6, Step 3:** Take a random sample of 50 recipients."
        ),
    )

    # Step 4: Create a dot chart
    ideology_chart = (
        alt.Chart(_sample)
        .mark_circle(size=100)
        .encode(
            x=alt.X("ideology_score:Q", title="Ideology Score"),    
            y=alt.Y("recipient_name:N", sort="x", title=None),
            color=alt.Color(
                "party:N",
                scale=alt.Scale(
                    domain=["Democrat", "Republican"], 
                    range=["#2166ac", "#b2182b"]
                ),
                title="Party"
            )
        )
        .properties(
            width=600, 
            height=600,
            title="Recipient Ideology Scores (Random Sample of 50)"
        )
    )  # YOUR CODE: alt.Chart(_sample).mark_circle(...)

    mo.stop(
        ideology_chart is ...,
        mo.md(
            "⚠️ **Complete Exercise 6, Step 4:** Create a dot chart showing recipient ideology scores colored by party."
        ),
    )

    mo.vstack(
        [
            ideology_chart,
            mo.md(
                "**Examine the chart:** Democrats and Republicans should cluster on opposite sides of zero."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### Exercise 7: Donor ideology distributions

    Now let's look at the **donor side**. Each donor's PC1 score is their
    estimated ideology.

    **Tasks:**

    1. Build a DataFrame of donor scores (provided below).
    2. Join with `contributors_meta` and `occ_industry` to get metadata (provided).
    3. Create a **faceted density plot** of ideology scores by **industry**
       (pick the top 6-8 industries by count, excluding "Other").

    *Hints:*

    - Use `transform_density("pc1_score", as_=["pc1_score", "density"], groupby=["industry"])`
      to compute densities, then `mark_area()`.
    - Facet with `row=alt.Row("industry:N", ...)` so each industry gets its own row.
    - To order by mean ideology, precompute a sorted list of industry names
      and pass it to `sort=` in `alt.Row()`.

    **Questions:**
    - Which industries are most polarized (wide spread)?
    - Which industries lean most liberal or conservative?
    """)
    return


@app.cell
def _(alt, contributors_meta, donor_ids, mo, np, occ_industry, pl, scores):
    # ---- Exercise 7 ----

    # Step 1 (provided): Build donor scores DataFrame
    _donor_scores = pl.DataFrame(
        {
            "bonica_cid": donor_ids,
            "pc1_score": scores[:, 0],
        }
    )

    # Step 2 (provided): Join with contributor metadata and industry mapping
    donor_ideology = (
        _donor_scores.join(contributors_meta, on="bonica_cid")
        .join(occ_industry, on="contributor_occupation", how="left")
        .with_columns(
            pl.col("industry").fill_null("Other"),
        )
    )

    # Clip extreme outliers for better visualization (provided)
    _p01 = np.percentile(donor_ideology["pc1_score"].to_numpy(), 1)
    _p99 = np.percentile(donor_ideology["pc1_score"].to_numpy(), 99)
    donor_plot = donor_ideology.filter(
        (pl.col("pc1_score") >= _p01) & (pl.col("pc1_score") <= _p99)
    )

    # Step 3: Faceted density plot by industry (top 8, excluding "Other")
    _top_8 = (
        donor_plot.filter(
            pl.col("industry") != "Other"
        )
        .group_by("industry")
        .count()
        .sort("count", descending=True)
        .head(8)
    )

    _industry_avg = (
        donor_plot.filter(
            pl.col("industry").is_in(_top_8["industry"])
        )
            .group_by("industry")
            .agg(pl.col("pc1_score").mean())
            .sort("pc1_score")["industry"]
            .to_list()
        )

    industry_chart = (
        alt.Chart(donor_plot.filter(pl.col("industry").is_in(_top_8["industry"])))
        .transform_density(
            "pc1_score",
            as_=["pc1_score", "density"],
            groupby=["industry"]
        )
        .mark_area()
        .encode(
            x=alt.X("pc1_score:Q", title="Ideology Score"),
            y=alt.Y("density:Q", stack=None, title=None, axis=None),

            row=alt.Row(
                    "industry:N", 
                    header=alt.Header(labelAngle=0, labelAlign='left'),
                    sort=_industry_avg,
                    title=None
                ),
                color=alt.Color("industry:N", legend=None)
            )
        .properties(width=400, height=60)
    )  # YOUR CODE

    mo.stop(
        industry_chart is ...,
        mo.md(
            "⚠️ **Complete Exercise 7, Step 3:** Create a faceted density plot of ideology scores by industry (top 8, excluding 'Other')."
        ),
    )

    industry_chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finance and business management are most polarized (widespread).
    Retirees leans most Republican.
    The education industry and unemployed lean most Democrat.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---

    ## Wrap-up

    In this lab you:

    1. **Explored** campaign finance data - who gives, who receives, and how
       donation flows differ by party and industry.
    2. **Normalized** a donor x recipient matrix to remove effects of donor
       prolificness and recipient popularity.
    3. **Ran PCA** on the normalized matrix to recover an ideological dimension.
    4. **Interpreted** the results by examining recipient loadings (which
       separate Democrats from Republicans) and donor ideology distributions
       (which reveal polarization patterns across industries).

    This is a simplified version of the method in:

    > Bonica, Adam. "Ideology and Interests in the Political Marketplace."
    > *American Journal of Political Science* 57, no. 2 (2013): 294-311.

    ### Submission

    Commit and push your completed notebook:

    ```bash
    git add notebooks/lab01/lab01.py
    git commit -m "Complete Lab 01"
    git push
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
