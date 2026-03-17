#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyQuake3D Plot Tool CLI

Command-line interface for visualizing and analyzing PyQuake3D simulations.
"""

import argparse

from pyquake_tools.plot_tool import Ptool


# -----------------------------------------------------------
# Argument parser
# -----------------------------------------------------------

def create_parser():

    parser = argparse.ArgumentParser(
        description="Visualization tool for PyQuake3D simulation outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the simulation directory",
    )

    # Operations
    parser.add_argument("--ts", action="store_true", help="Plot time series")
    parser.add_argument("--a2", action="store_true", help="Create 2D animation")
    parser.add_argument("--a3", action="store_true", help="Create 3D animation")
    parser.add_argument("--event", action="store_true", help="Extract earthquake events")
    parser.add_argument("--phase", action="store_true", help="Generates phase plot (V vs Psi)")
    parser.add_argument("--event_plot", action="store_true", help="Generates rupture prop. in xvs time")

    # Visualization parameters
    parser.add_argument("--var", type=str, help="Variable to plot (V, Omega, state)")
    parser.add_argument("--t_min", type=float, help="Minimum time")
    parser.add_argument("--t_max", type=float, help="Maximum time")

    parser.add_argument("--V_min", type=float)
    parser.add_argument("--V_max", type=float)

    parser.add_argument("--Omega_min", type=float)
    parser.add_argument("--Omega_max", type=float)

    parser.add_argument("--theta_min", type=float)
    parser.add_argument("--theta_max", type=float)

    parser.add_argument("--azimuth", type=float)
    parser.add_argument("--elevation", type=float)

    parser.add_argument("--interval", type=int)
    
    parser.add_argument("--event_no", type=int, help ='Required for event_plot')
    parser.add_argument("--depth",  type=float, help ='Required for event_plot')

    # Physics parameters
    parser.add_argument("--V_dyn", type=float, help ='Dynamic slip rate')
    parser.add_argument("--G", type=float, help ='Shear modulus')
    parser.add_argument("--rho", type=float, help ='Rock density')

    return parser


# -----------------------------------------------------------
# Apply CLI parameters
# -----------------------------------------------------------

def apply_parameters(tool, args):

    parameters = [
        "var",
        "t_min",
        "t_max",
        "V_min",
        "V_max",
        "Omega_min",
        "Omega_max",
        "theta_min",
        "theta_max",
        "azimuth",
        "elevation",
        "interval",
        "V_dyn",
        "G",
        "rho",
        "depth",
        "event_no"
    ]

    for param in parameters:

        value = getattr(args, param)

        if value is not None:
            setattr(tool, param, value)


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------

def main():

    parser = create_parser()
    args = parser.parse_args()

    tool = Ptool(args.path)

    apply_parameters(tool, args)

    if args.ts:
        tool.plot_timeseries()

    if args.event:
        tool.extract_slip_info()

    if args.a2:
        tool.animation2D()

    if args.a3:
        tool.animation3D()
        
    if args.phase:
        tool.phase_plot()

    if args.event_plot:
        tool.event_plot()

if __name__ == "__main__":
    main()
