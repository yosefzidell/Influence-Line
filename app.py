import dash
from dash import dcc, html
from dash.dependencies import Input, Output, MATCH, ALL
import dash.exceptions
import dash_bootstrap_components as dbc
import matplotlib
from matplotlib.pyplot import colorbar

matplotlib.use('Agg')

import numpy as np
import plotly.graph_objs as go
import gc


#app = dash.Dash(__name__, suppress_callback_exceptions=True)
app = dash.Dash(__name__,external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
server = app.server  # For deployment if needed



# Helper function to create the Matplotlib figure
def create_beam_figure(n_supports,spacings,left,right,location,influence,hover_data):





    # =====================
    # Update Beam Diagram
    # =====================
    beam_length = sum(spacings)  # Total length based on number of supports and spacing
    beam_x = [0, beam_length]
    beam_y = [0, 0]

    # Beam line
    beam_trace = go.Scatter(
        x=beam_x,
        y=beam_y,
        mode='lines',
        line=dict(color='#000080', width=5),
        name='Beam',
    )

    # The supports are all going to be simple supports except potentially the end supports.
    support_types = ['Simple', 'Moment', 'None (Overhang)']
    # Supports as triangles
    support_traces = []
    support_width = .5  # width of the base of the triangle
    support_height = .5  # height of the triangle

    # Calculate positions of supports relative to left end
    support_positions = []
    current=0
    for i in spacings:
        current += i
        support_positions.append(current)


    #add supports to beam
    for idx, sx in enumerate(support_positions):
        if sx==0: #begin support
            if left==support_types[0]: #left end support is simple (triangle support)
                support_traces.append(go.Scatter(
                    x=[sx - support_width / 2, sx + support_width / 2, sx,sx - support_width / 2],
                    y=[-support_height,-support_height,0,-support_height],
                    mode='lines',
                    fill='toself',
                    fillcolor='#8B0000',
                    line=dict(color='black'),
                    name='Support',
                    hoverinfo='none'  # No hover info for supports
                ))
            elif left==support_types[1]: #left end support is moment (rectangle support)
                support_traces.append(go.Scatter(
                    x=[sx - support_width / 4, sx, sx,sx - support_width / 4,sx - support_width / 4],
                    y=[+support_height,+support_height, -support_height,-support_height,+support_height],
                    mode='lines',
                    fill='toself',
                    fillcolor='#8B0000',
                    line=dict(color='black'),
                    name='Support',
                    hoverinfo='none'  # No hover info for supports
                ))

        elif sx==beam_length: #end support
            if right==support_types[0]: #right end support is simple (triangle support)
                support_traces.append(go.Scatter(
                    x=[sx - support_width / 2, sx + support_width / 2, sx,sx - support_width / 2],
                    y=[-support_height,-support_height,0,-support_height],
                    mode='lines',
                    fill='toself',
                    fillcolor='#8B0000',
                    line=dict(color='black'),
                    name='Support',
                    hoverinfo='none'  # No hover info for supports
                ))
            elif right==support_types[1]: #right end support is moment (rectangle support)
                support_traces.append(go.Scatter(
                    x=[sx + support_width / 4, sx, sx,sx + support_width / 4,sx + support_width / 4],
                    y=[+support_height,+support_height, -support_height,-support_height,+support_height],
                    mode='lines',
                    fill='toself',
                    fillcolor='#8B0000',
                    line=dict(color='black'),
                    name='Support',
                    hoverinfo='none'  # No hover info for supports
                ))

        else: #all  middle supports
            support_traces.append(go.Scatter(
                x=[sx - support_width / 2, sx + support_width / 2, sx, sx - support_width / 2],
                y=[-support_height, -support_height, 0, -support_height],
                mode='lines',
                fill='toself',
                fillcolor='#8B0000',
                line=dict(color='black'),
                name='Support',
                hoverinfo='none'  # No hover info for supports
            ))



    # Supports Labels
    SUPPORT_LABELS = list("ABCDEFGHIJ")  # Labels from A to J for supports 1 to 10
    label_traces = []
    for idx, sx in enumerate(support_positions):
        label = SUPPORT_LABELS[idx]


        # Position the label slightly below the support
        label_x = sx
        label_y = -support_height - 0.25  # Adjust the y-position as needed

        label_traces.append(go.Scatter(
            x=[label_x],
            y=[label_y],
            mode='text',
            text=f'<b>{label}<b>',
            textposition='top center',
            textfont=dict(size=12, color='black'),
            showlegend=False,
            hoverinfo='none'
        ))

    # Add two-sided arrows between supports
    annotations = []
    for i,j in zip(range(len(support_positions) - 1),range(len(spacings)-1)):
        start_x = support_positions[i]
        end_x = support_positions[i + 1]
        mid_x = (start_x + end_x) / 2  # Middle point for the text

        #create a two-way arrow that shows the spacing between supports
        # Create annotation for left to right arrow
        annotations.append(dict(
            ax=start_x+.25, ay=-.7,  # Start point of the arrow
            x=end_x-.25, y=-.7,  # End point of the arrow
            xref='x', yref='y',
            showarrow=True,
            arrowhead=2,  # Two-sided arrow
            arrowsize=1.5,
            arrowwidth=1.5,
            axref='x', ayref='y',
            standoff=4
        ))
        # Create annotation for right to left arrow
        annotations.append(dict(
            ax=end_x-.25, ay=-.7,  # Start point of the arrow
            x=start_x+.25, y=-.7,  # End point of the arrow
            xref='x', yref='y',
            showarrow=True,
            arrowhead=2,  # Two-sided arrow
            arrowsize=1.5,
            arrowwidth=1.5,
            axref='x', ayref='y',
            standoff=4
        ))

        # Text "j ft" at the center of the arrow
        annotations.append(dict(
            x=mid_x, y=-1,  # Position text slightly below the arrow
            xref='x', yref='y',
            text=f'<b>{spacings[j+1]} ft<b>',
            showarrow=False,
            font=dict(size=12, color='black')
        ))
    # Check if user is hovering over the second graph (xy-graph)
    if hover_data is not None:
        hover_x = hover_data['points'][0]['x']
        # Add a vertical arrow at the hovered x position
        annotations.append(dict(
            x=hover_x,
            y=0.1,
            xref="x",
            yref="y",
            text="P=1",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40,
            arrowcolor="red",
            arrowwidth=2
        ))

    # Add a vertical dashed line at slider location on beam
    influence_section = go.Scatter(
        x=[location, location],
        y=[1, -1],
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name='x = 0 (vertical line)')

    # Combine the beam and support traces
    beam_fig = go.Figure(data=[beam_trace] + support_traces + label_traces+ [influence_section])

    beam_fig.update_layout(
        xaxis=dict(title='Length (ft)', range=[-1, beam_length + 1],
                   zeroline=False, showgrid=False),
        yaxis=dict(title='', range=[-support_height - 1, 2], showticklabels=False,
                   zeroline=False, showgrid=False),
        showlegend=False,
        height=300,
        margin=dict(l=0, r=0, t=0, b=0),
        annotations = annotations
    )


    # =====================
    # Update Equation Graph
    # =====================
    analysis_type = ['Shear Influence', 'Moment Influence']
    error_message = []

    x = np.arange(0, beam_length, .01)
    if n_supports == 2:
        if left == support_types[0] and right == support_types[0]:

            L = sum(spacings)
            c = location
            A_y = [i / L for i in x]
            if influence == analysis_type[0]:  # shear influence line selected
                y = [-i if j < c else -i + 1 for i, j in zip(A_y, x)]
            else:  # moment influence line selected
                y = [(i) * (L - c) / L if i <= c else (L - c) * c / L - (i - c) * ((L - c) * c / L) / (L - c) for i in
                     x]

        elif left == support_types[1] and right == support_types[0]:
            # one end is a moment connection and the other is a simple support:

            # remove support B and solve its deflection
            L = sum(spacings)
            c = location
            delta_Bx = np.array([i ** 2 / 6 * (3 * L - i) for i in x])
            # solve reaction By with adding the support back in thats equal to delta_Bx
            B_y = np.array([i * 3 / (L ** 3) for i in delta_Bx])
            # now that B_y is solved, solve M_A and A_y using equilibrium:
            M_A = np.array([-i * L + j for i, j in zip(B_y, x)])
            A_y = np.array([(i + L - j) / L for i, j in zip(M_A, x)])

            # now we can determine the shear influence using A_y
            y = np.array([i - 1 if j < c else i for i, j in zip(A_y, x)])

        elif left == support_types[0] and right == support_types[1]:
            # one end is a moment connection and the other is a simple support:

            # remove support A and solve its deflection
            L = sum(spacings)
            c = location
            delta_Ax = np.array([(L - i) ** 2 / 6 * (3 * L - (L - i)) for i in x])
            # solve reaction By with adding the support back in thats equal to delta_Bx
            A_y = np.array([i * 3 / (L ** 3) for i in delta_Ax])
            # now that B_y is solved, solve M_A and A_y using equilibrium:
            M_B = np.array([-i * L + (L - j) for i, j in zip(A_y, x)])
            B_y = np.array([(i + j) / L for i, j in zip(M_B, x)])

            y = np.array([i - 1 if j < c else i for i, j in zip(A_y, x)])


        elif left == support_types[1] and right == support_types[1]:
            # both ends are moments supports

            # solve for right side support Ay and M_A
            L = beam_length
            c = location

            A_y = np.array([((L - i) ** 2 / (L ** 3)) * (3 * i + (L - i)) for i in x])
            M_A = np.array([(i * (L - i) ** 2) / (L ** 2) for i in x])

            # now we can determine the shear influence using A_y and M_A
            # now we can determine the influence using A_y and M_A
            if influence == analysis_type[0]:  # shear influence line selected
                y = [i - 1 if j <= c else i for i, j in zip(A_y, x)]
            else:  # moment influence line selected
                y = [i * c - k - (c - j) if j <= c else i * c - k for i, j, k in zip(A_y, x, M_A)]

        elif left == support_types[1] and right == support_types[2]:
            # cantilevered with moment support on left

            # solve for right side support Ay and M_A
            L = beam_length
            c = location

            A_y = np.array([1 for _ in x])
            M_A = np.array([i for i in x])

            # now we can determine the shear influence using A_y and M_A
            # now we can determine the influence using A_y and M_A
            if influence == analysis_type[0]:  # shear influence line selected
                y = [i - 1 if j <= c else i for i, j in zip(A_y, x)]
            else:  # moment influence line selected
                y = [i * c - k - (c - j) if j <= c else i * c - k for i, j, k in zip(A_y, x, M_A)]

        elif left == support_types[2] and right == support_types[1]:
            # cantilevered with moment support on right

            # solve for right side support Ay and M_A
            L = beam_length
            c = location

            # determine influence with no support values needed
            if influence == analysis_type[0]:  # shear influence line selected
                y = [1 if i <= c else 0 for i in x]
            else:  # moment influence line selected
                y = [c - i if i <= c else 0 for i in x]
        else:

            L = beam_length  # total length of beam
            c =location  # location of interest for influence line diagram, determined by the slider
            y = [1 for i in x]


    elif n_supports == 3:

        L = beam_length  # total length of beam
        c = location  # location of interest for influence line diagram, determined by the slider
        L1 = spacings[1]  # distance of middle support from left end
        L2 = spacings[2]  # distance of middle support from right end

        if left == support_types[0] and right == support_types[0]:
            # a beam with 3 or more supports is considered statically indeterminant, thus making the classic equilibrium
            # equations insufficient for any shear calculations. rather, before any influence line equations can be derived,
            # we must first solve a reaction in terms of the load location "x" and then can continue
            # with our new adjusted statically "determinant" structure.

            # we will solve the reaction(x) of the middle support by superposition (removing the support
            # and solving the deflection at the middle support location, then solving the required P load at
            # the middle support to produce the same upwards deflection. the following list determines the
            # deflection of the middle support when the load is applied at every position "a" on the beam

            delta_Bx = np.array(
                [(L - i) * L1 / (6 * L) * (L ** 2 - (L - i) ** 2 - L1 ** 2) if i >= L1 else i * L2 / (6 * L) * (
                        L ** 2 - i ** 2 - L2 ** 2) for i in x])
            # this is deflection of middle support as a function of the load location

            # delta_BB= P(L1^2)(L2^2)/3L , this is the deflection @ middle support when upward P load applied at the middle support
            # equating both equations allows us to solve for P (shear @influence location when load placed there):
            B_y = np.array([i / ((L1 ** 2) * (L2 ** 2) / (3 * L)) for i in delta_Bx])

            # Now that the middle reaction is determined, the other supports can be solved as functions of load position "a" as well.

            A_y = np.array([(-i * L2 + (L - j)) / L for i, j in zip(B_y, x)])
            C_y = np.array([(-i * L1 + j) / L for i, j in zip(B_y, x)])

            # so far we have created lists for the reactions of A, B , and C when the load is placed on the beam
            # we will use these reaction values to determine the positive and negative influence value at the slider location:
            # using simple statics:

            # I_loc_neg=np.array([location,A_y[np.where(x == location)[0][0]]-1])
            # I_loc_pos=np.array([location,A_y[np.where(x == location)[0][0]]])

            # now we can add the remaining known Influence values, which is when the load is placed at the supports:
            # if location == 0:
            #     I_Ay=np.array([0,1])
            #     I_By=np.array([spacings[0],0])
            #     I_Cy = np.array([spacings[0]+spacings[1],0])
            # if location==spacings[0]:
            #     I_Ay = np.array([0, 0])
            #     I_By = np.array([spacings[0],1])
            #     I_Cy = np.array([spacings[0] + spacings[1], 0])
            #  if location==spacings[0]+spacings[1]:
            #     I_Ay = np.array([0, 0])
            #     I_By = np.array([spacings[0],0])
            #     I_Cy = np.array([spacings[0] + spacings[1], 1])
            #
            # else:
            #     I_Ay = np.array([0, 0])
            #     I_By = np.array([spacings[0], 0])
            #     I_Cy = np.array([spacings[0] + spacings[1], 0])

            # Now that we have the Ay and By reactions for all load positions, we can determine the influence value
            # for "location"
            if influence == analysis_type[0]:  # shear influence line selected
                if c < L1:
                    y = np.array([i - 1 if j < c else i for i, j in zip(A_y, x)])
                elif c == L1:
                    y = np.array([i for i in B_y])
                else:
                    y = np.array([(i + k - 1) if j < c else (i + k) for i, k, j in zip(A_y, B_y, x)])
            else:  # moment influence line selected
                if c <= L1:
                    y = np.array([i * c - (c - j) if j < c else i * c for i, j in zip(A_y, x)])
                else:
                    y = np.array(
                        [i * c - (c - j) + k * (c - L1) if j < c else i * c + k * (c - L1) for i, j, k in
                         zip(A_y, x, B_y)])
        elif left == support_types[0] and right == support_types[1]:
            # Moment support on right
            # determine middle support By using its redundant deflection
            delta_Bx = np.array(
                [(L - i) ** 2 * L1 / (12 * L ** 3) * (
                        3 * i * L ** 2 - 2 * L * L1 ** 2 - i * L1 ** 2) if i >= L1 else i / (12 * L ** 3) * (
                        L2 ** 2) * (
                                                                                                3 * L ** 2 * L1 - L1 * i ** 2 - 2 * L * i ** 2)
                 for i in x])
            B_y = np.array([(i * 12 * L ** 3) / ((L1 ** 2 * L2 ** 3) * (3 * L + L1)) for i in delta_Bx])

            # solve left support A_y by using its redundant deflection. remove support A, solve deflection at
            # free end from Unit Load and B_y, and solve A_y by making it equal that deflection.

            delta_B_cantilever = np.array(
                [L2 ** 2 / 6 * (3 * (L - i) - L + L1) if i <= L1 else (L - i) ** 2 / 6 * (3 * L - 3 * L1 - (L - i)) for
                 i in x])
            # B_y_cantilever=np.array([3*i/((L-j)**3) for i,j in zip(delta_B_cantilever,x)])
            delta_A1 = np.array([(L - i) ** 2 / 6 * (3 * L - (L - i)) for i in x])
            delta_A2 = np.array([i * (L2 ** 2) / 6 * (3 * L - L2) for i in B_y])
            A_y = np.array([3 * (i - j) / (L ** 3) for i, j in zip(delta_A1, delta_A2)])
            # now y can be determined by using A_y and B_y
            if influence == analysis_type[0]:  # shear influence line selected
                if c < L1:
                    y = np.array([i - 1 if j < c else i for i, j in zip(A_y, x)])
                elif c == L1:
                    y = np.array([i for i in B_y])
                else:
                    y = np.array([(i + k - 1) if j < c else (i + k) for i, k, j in zip(A_y, B_y, x)])
            else:  # moment influence line selected
                if c <= L1:
                    y = np.array([i * c - (c - j) if j < c else i * c for i, j in zip(A_y, x)])
                else:
                    y = np.array([i * c - (c - j) + k * (c - L1) if j < c else i * c + k * (c - L1) for i, j, k in
                                  zip(A_y, x, B_y)])
        elif left == support_types[1] and right == support_types[0]:
            # Moment support on left
            # the influence line will mirror the previous condition , thus the same procedure will be used by changing x to (L-x) and L1 to L2, and vice versa
            # determine middle support By using its redundant deflection
            delta_Bx = np.array(
                [(i) ** 2 * L2 / (12 * L ** 3) * (3 * (L - i) * L ** 2 - 2 * L * L2 ** 2 - (L - i) * L2 ** 2) if (
                                                                                                                         L - i) > L2 else (
                                                                                                                                                  L - i) / (
                                                                                                                                                  12 * L ** 3) * (
                                                                                                                                                  L1 ** 2) * (
                                                                                                                                                  3 * L ** 2 * L2 - L2 * (
                                                                                                                                                  L - i) ** 2 - 2 * L * (
                                                                                                                                                          L - i) ** 2)
                 for i in x])
            B_y = np.array([(i * 12 * L ** 3) / ((L2 ** 2 * L1 ** 3) * (3 * L + L2)) for i in delta_Bx])

            # solve right support A_y by using its redundant deflection. remove support A, solve deflection at
            # free end from Unit Load and B_y, and solve A_y by making it equal that deflection.

            delta_B_cantilever = np.array(
                [L2 ** 2 / 6 * (3 * (L - i) - L + L1) if i <= L1 else (L - i) ** 2 / 6 * (3 * L - 3 * L1 - (L - i)) for
                 i in x])
            # B_y_cantilever=np.array([3*i/((L-j)**3) for i,j in zip(delta_B_cantilever,x)])
            delta_A1 = np.array([(i) ** 2 / 6 * (3 * L - (i)) for i in x])
            delta_A2 = np.array([i * (L1 ** 2) / 6 * (3 * L - L1) for i in B_y])
            A_y = np.array([3 * (i - j) / (L ** 3) for i, j in zip(delta_A1, delta_A2)])
            # now y can be determined by using A_y and B_y
            if influence == analysis_type[0]:  # shear influence line selected
                if (L - c) < L2:
                    y = np.array([i - 1 if (L - j) < (L - c) else i for i, j in zip(A_y, x)])
                elif (L - c) == L2:
                    y = np.array([i for i in B_y])
                else:
                    y = np.array([(i + k - 1) if (L - j) < (L - c) else (i + k) for i, k, j in zip(A_y, B_y, x)])
            else:  # moment influence line selected
                if (L - c) <= L2:
                    y = np.array([i * (L - c) - ((L - c) - (L - j)) if (L - j) < (L - c) else i * (L - c) for i, j in
                                  zip(A_y, x)])
                else:
                    y = np.array([i * (L - c) - ((L - c) - (L - j)) + k * ((L - c) - L2) if (L - j) < (L - c) else i * (
                            L - c) + k * ((L - c) - L2) for i, j, k in zip(A_y, x, B_y)])
        elif left == support_types[1] and right == support_types[1]:
            # both ends are moments supports

            delta_Bx = np.array(
                [(L - i) ** 2 * L1 ** 2 / (6 * L ** 3) * (3 * i * L - 3 * L1 * i - L1 * (L - i)) if i > L1 else (
                                                                                                                    i) ** 2 * L2 ** 2 / (
                                                                                                                        6 * L ** 3) * (
                                                                                                                        3 * (
                                                                                                                        L - i) * L - 3 * (
                                                                                                                                L - i) * L2 - i * L2)
                 for i in x])
            B_y = np.array([(i * 3 * L ** 3) / (L2 ** 3 * L1 ** 3) for i in delta_Bx])

            # now solve for M_A and A_y by determining M_A an A_y from the effects of the unit load, and B_y individually.

            A_y1 = np.array([((L - i) ** 2 / (L ** 3)) * (3 * i + (L - i)) for i in x])  # A_y from unit load
            M_A1 = np.array([(i * (L - i) ** 2) / (L ** 2) for i in x])  # M_A from unit load

            A_y2 = np.array([(i * (L2) ** 2 / (L ** 3)) * (3 * L1 + L2) for i in B_y])  # A_y from B_y
            M_A2 = np.array([(i * (L1) * (L2) ** 2) / (L ** 2) for i in B_y])  # M_A from B_y

            A_y = np.array([i - j for i, j in zip(A_y1, A_y2)])
            M_A = np.array([i - j for i, j in zip(M_A2, M_A1)])

            # now we can determine the influence using A_y and M_A
            if influence == analysis_type[0]:  # shear influence line selected
                if c == 0:
                    y = np.array([i for i in A_y])
                if c < L1:
                    y = np.array([i - 1 if j < c else i for i, j in zip(A_y, x)])
                elif c == L1:
                    y = np.array([i for i in B_y])
                else:
                    y = np.array([(i + k - 1) if j < c else (i + k) for i, k, j in zip(A_y, B_y, x)])
            else:  # moment influence line selected
                if c <= L1:
                    y = np.array([i * c + k - (c - j) if j < c else i * c + k for i, j, k in zip(A_y, x, M_A)])
                else:
                    y = np.array(
                        [i * c + k - (c - j) + b * (c - L1) if j < c else i * c + k + b * (c - L1) for i, j, k, b in
                         zip(A_y, x, M_A, B_y)])
        elif left == support_types[2] and right == support_types[0]:
            # Overhang on left side, simple support on right
            # with just two simple supports, By and Cy can be determined with simple statics
            # Moment about support C is zero, thus:
            B_y = np.array([(L - i) / L2 for i in x])

            # now solve for shear and moment influence:
            if influence == analysis_type[0]:  # shear influence line selected
                if c < L1:
                    y = np.array([-1 if i < c else 0 for i in x])
                elif c == L1:
                    y = np.array([i for i in B_y])
                else:
                    y = np.array([(i - 1) if j < c else i for i, j in zip(B_y, x)])
            else:  # moment influence line selected
                if c <= L1:
                    y = np.array([-(c - i) if i < c else 0 for i in x])
                else:
                    y = np.array(
                        [-(c - i) + b * (c - L1) if i < c else b * (c - L1) for i, b in zip(x, B_y)])
        elif left == support_types[0] and right == support_types[2]:
            # Overhang on right side, simple support on left
            # with just two simple supports, By and Cy can be determined with simple statics
            # Moment about support A is zero, thus:
            B_y = np.array([i / L1 for i in x])
            #now solve for Ay
            A_y=np.array([1-j for j in B_y])

            # now solve for shear and moment influence:
            if influence == analysis_type[0]:  # shear influence line selected
                if c > L1:
                    y = np.array([-1 if i > c else 0 for i in x])
                elif c == L1:
                    y = np.array([i for i in B_y])
                else:
                    y = np.array([(i - 1) if j < c else i for i, j in zip(A_y, x)])
            else:  # moment influence line selected
                if c >= L1:
                    y = np.array([-(c - i) if i > c else 0 for i in x])
                else:
                    y = np.array(
                        [-(c - i) + b * (c) if i < c else b * (c) for i, b in zip(x, A_y)])


    # PLACEHOLDER FOR SUPPORTS 4 AND 5
    else: # n_supports > 3:
        x = np.arange(0, beam_length, .01)
        y = [0 for i in x]



    equation_trace = go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name=f'Influence Chart',
        line=dict(color='#000080', width=3),
        hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>'
    )
    zero_line=go.Scatter(
    x=[0, beam_length],
    y=[0, 0],
    mode='lines',
    line=dict(color='red', width=2, dash='dash'),
    name='y = 0 (horizontal line)')






    # Create the equation figure
    equation_fig = go.Figure(data=[equation_trace]+[zero_line])
    if n_supports>3:
        equation_fig.add_annotation(x=beam_length/2,  # X-coordinate
    y=.5,  # Y-coordinate
    text="Influence Line not available yet. try again soon!",  # The text to display
    font=dict(size=36, color='black'),

     )
    else:
        pass

    equation_fig.update_layout(
        xaxis=dict(title='unit load position on beam', range=[0, beam_length]),
        yaxis=dict(title='Influence Value',range=[min(y)-1,max(y)+1]),
        showlegend=False,
        hovermode='closest',
        margin=dict(l=0, r=0, t=0, b=0)
    )

    #update slider values
    slider_max = beam_length # Slider max is n_supports * beam_length
    slider_marks = {i: str(i) for i in range(0, slider_max + 1)}

    return beam_fig, equation_fig, slider_max, slider_marks




# Layout for the Dash app
app.layout = html.Div([
    html.Div([html.H3('Influence Line Creator')],style={'display': 'flex', 'align-items': 'center','justify-content': 'center', 'margin-bottom': '20px'}),
    html.Div([
        html.Label('Select number of supports:', style={'margin-right': '10px'}),
        dcc.Dropdown(
            id='support-dropdown',
            options=[{'label': str(i), 'value': i} for i in range(2, 11)],
            value=2,  # Default value
            clearable=False,
            style={'width': '120px'}
        )
    ], style={'display': 'flex', 'align-items': 'center','justify-content': 'center', 'margin-bottom': '20px'}),
    html.Div([
            html.Label('Select Influence Type:', style={'margin-right': '10px'}),
            dcc.Dropdown(
                id='Influence-dropdown',
                options=['Shear Influence','Moment Influence'],
                value='Shear Influence',  # Default selection
                clearable=False,
                style={'width': '200px'}
            )
        ], style={'display': 'flex', 'align-items': 'center','justify-content': 'center', 'margin-bottom': '20px'}),


# Container for dynamically generated input fields for custom spacing
    html.Div(id='input-boxes', style={'margin-bottom': '20px'}),

html.Br(),
html.Br(),

# Dropdown for far left support type
html.Div(
    children=[
        html.Div(
            children=[
                # Left-aligned label and dropdown
                html.Div(
                    children=[
                        html.H6("Left End Support Type:", style={'margin-left': '210px'}),
                        html.H6("Right End Support Type:", style={'margin-right': '200px'}),

                    ],
                    style={'display': 'flex', 'align-items': 'center','justify-content': 'space-between','width': '100%','padding': '5px'}  # Flexbox for left items
                ),
                # dropdowns and beam diagram label
                html.Div(
                    children=[
                        dcc.Dropdown(
                            id='left-dropdown',
                            options=['Simple', 'Moment', 'None (Overhang)'

                                     ],
                            value='Simple',
                            style={'width': '200px','margin-left': '105px'}
                        ),
                        html.H5('Beam Diagram:', style={'margin-left': '100px'}),
                        dcc.Dropdown(
                            id='right-dropdown',
                            options=['Simple', 'Moment', 'None (Overhang)'

                                     ],
                            value='Simple',
                            style={'width': '200px','margin-right': '190px'}
                        ),
                    ],
                    style={'display': 'flex', 'align-items': 'center','justify-content': 'space-between','width': '100%','padding': '5px',}
                ),
            ],


        ),
    ]
),

    html.Div([
        dcc.Graph(id='beam-graph')
    ], style={'width': '80%', 'margin': 'auto'}),

html.Br(),

# Slider to select influence position

html.Div([
            html.H6("Select the influence position by dragging slider below:",style={'margin-left': '20px'}),
            dcc.Slider(
                id='slider_x',
                min=0,
                max=5,  # Will be updated based on Length of beam
                step=.1,
                value=0,
                marks={i: str(i) for i in range(0,6)},
                updatemode='drag',  # Update value while dragging


    )],style={'width': '70%', 'margin': 'auto','margin-left':'300px'}),
# Container for Influence Chart Label
    html.Div(id='influence-label', style={'margin-top': '10px','align-items': 'center','justify-content': 'center'}),
html.Br(),

    html.Div([
        dcc.Graph(id='influence-graph',config={'displayModeBar': False})
    ], style={'width': '80%', 'margin': 'auto','padding':'0'}),
html.Br(),
html.Br()





])

@app.callback(
    Output('input-boxes','children'),
    Input('support-dropdown', 'value'),

        )

def update_spacing_inputs(n_supports):
    inputs = []
    # Create input fields for the spacing between each pair of supports
    for i in range(n_supports - 1):
        inputs.append(html.Div([
            html.Label(f'Spacing between support {chr(65 + i)} and {chr(65 + i + 1)}:', style={'margin-right': '10px'}),
            dcc.Input(
                id={'type': 'input-box', 'index': i + 1},
                type='number',
                value=5,  # Default value is 5 ft
                min=0,
                style={'width': '100px', 'margin-right': '20px'}
            )
        ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}))
    return inputs

@app.callback(
    Output('influence-label','children'),
    [Input('Influence_dropdown', 'value'),
     Input('slider-x', 'value')]


        )
def update_influence_label(influence_type,location):
      return f' {influence_type} @ x= {location} ft'




#callback to update beam diagram, influence diagram, and slider
@app.callback(
    [Output('beam-graph', 'figure'),
     Output('influence-graph', 'figure'),
     Output('slider_x', 'max'),
     Output('slider_x', 'marks')],
    [Input('support-dropdown', 'value'),
     Input({'type': 'input-box', 'index': ALL}, 'value'),
     Input('left-dropdown','value'),
     Input('right-dropdown','value'),
     Input('slider_x', 'value'),
     Input('Influence-dropdown', 'value'),
     Input('influence-graph', 'hoverData')],
)
def update_beam(n_supports,values,left,right,location,influence,hover_data):
    # create a list of current support spacings

    spacings = [i if i is not None else 0 for i in values]
    spacings.insert(0,0)
    gc.collect()
    return create_beam_figure(n_supports,spacings,left,right,location,influence,hover_data)








# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)

