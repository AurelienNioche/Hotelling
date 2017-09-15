from os import path


an_parameters = {
    "working_folder": path.expanduser("~/Desktop/HotellingExperimental/data"),
    "fig_folder": path.expanduser("~/Desktop/HotellingExperimental/figures2"),

    "t_max": 5000,
    "time_window": 100,
    "customer_firm_choices_period": 25,
    "firm_period": 5000,

    "display": False,
    "linear_regression": True,

    "scatter_vars": [
        # ("firm_temp", "delta_position"),
        # ("firm_alpha", "delta_position"),
        # ("firm_temp", "delta_price"),
        # ("firm_alpha", "delta_price"),
        # ("customer_temp", "delta_position"),
        # ("customer_alpha", "delta_position"),
        # ("customer_temp", "delta_price"),
        ("customer_extra_view_choices", "delta_position"),
        # ("customer_alpha", "delta_price"),
        # ("customer_extra_view_choices", "profits"),
        # ("customer_extra_view_choices", "delta_price"),
        # ("customer_utility_consumption", "delta_price"),
        # ("customer_utility_consumption", "delta_position"),
        # ("customer_utility_consumption", "customer_extra_view_choices"),
        ("transportation_cost", "customer_extra_view_choices"),
        # ("transportation_cost", "delta_position"),
        # ("transportation_cost", "delta_price"),
        # ("transportation_cost", "profits"),
        # ("transportation_cost", "customer_utility"),
        ("delta_position", "profits"),
        # ("delta_position", "customer_utility"),
        # ("delta_position", "delta_price"),
        # ("profits", "customer_utility")
    ],

    "curve_vars": ["profits", "customer_utility"],

    "range_var": {
        "firm_temp": (0.0095, 0.0305),
        "customer_temp": (0.0095, 0.0305),
        "firm_alpha": (0.009, 0.101),
        "customer_alpha": (0.009, 0.101),
        "transportation_cost": (-0.01, 1.01),
        "delta_price": (-0.1, 8.1),
        "delta_position": (-0.1, 8.1),
        "profits": (9, 62.1),
        "customer_utility": (-0.1, 20.1),
        "customer_utility_consumption": (11.9, 23.1),
        "customer_extra_view_choices": (-0.1, 10.1)
    },
}

