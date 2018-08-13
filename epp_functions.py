
# ------------------- IMPORTANT NOTICE -------------------------------------- #
#
# THE ALGORITHMS IN THIS FILE DO NOT MATCH EXACTLY THE ONES PRESENTED IN THE 
# PUBLICATION. IN THE PUBLICATION, THE ALGORITHMS ARE PRESENTED IN THEIR 
# SIMPLEST FORM FOR CLARITY. HERE, THEY ALGORITMS ARE MODIFIED TO TAKE BETTER 
# ADVANTAGE OF THE TEST FUNCTION.
#
# ------------------- IMPORTANT NOTICE -------------------------------------- #


from smallestCircle import *
import numpy as np
import pandas as pd
import nvector as nv
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

M = Basemap(projection='merc')
TO_RADIANS = np.pi/180
TO_DEGREES = 180/np.pi

def digest_messages(messages):
    """
    Takes a pandas dataframe containing AIS data and returns a list of 
    trajectories. Each trajectory is a list of (longitude,latitude,
    timestamp) tuples. In each trajectory, the timestep difference 
    between successive tuples does not exceed 5 minutes. 
    """
    trajectories = []
    # Remove duplicate messages
    messages = messages.drop_duplicates(subset='timestamp',keep='first') \
        .sort_values(by='timestamp') \
        .rename(columns = {'longitude':'lon','latitude':'lat'})
    # Add column with time between messages (dt)
    messages['dt'] = pd.Series(
        np.diff(messages.timestamp).astype('timedelta64[m]'),
        index=messages.index[1:]
        )
    # Split messages into groups at time gaps ( dt > 5 min )
    grouped_messages = np.split(messages, np.where(messages.dt > pd.Timedelta('5 min'))[0][1:])
    # Draw trajectories from time-grouped messages and segment them
    # by basic behaviours
    for group in grouped_messages:
        subset = group[['lon', 'lat', 'timestamp']]
        trajectory = [tuple(x) for x in subset.values]
        trajectories.append(trajectory)
    return trajectories

def epp_simplify(trajectory):
    ss = segment(trajectory)
    st = translate(ss)
    return st

def segment(trajectory):
    """
    Input: list of (longitude,latitude,timestamp) tuples.
    Output: list of dicts containing data of the basic behaviour segments.
    """
    segments = []
    current_t = trajectory
    while True:        
        split, result_t, data_t = split_fcn(current_t)
        segments.append(data_t)
        if split:            
            current_t = result_t
        else:
            break
    return segments

def split_fcn(trajectory):
    """
    Input: list of (longitude,latitude,timestamp) tuples.
    """
    split = False
    first = 1
    last = len(trajectory)
    while first <= last and not split:
        mid = (first + last)//2
        d, d_data = test(trajectory[:mid])
        u, u_data = test(trajectory[:mid+1])
        if d and not u:
            split = True
        else:
            if d and u:
                first = mid + 1
            else:
                last = mid - 1
    if split:
        return True, trajectory[mid-1:], d_data
    else:
        return False, trajectory, u_data

def test(trajectory,dc=3):

    label = None
    traj_len = len(trajectory)

    if traj_len == 1:
        label = 'crumb'
    else:
        
        # Check if any two vertices have the same longitude and latitude
        lon, lat, time = zip(*trajectory)
        points = list(zip(lon,lat))
        if len(set(points)) != len(trajectory):
            identical = True
        else:
            identical = False
        
        # Determine if the trajectory can be enclosed by a circle of 100 meters
        # diameter or less. 
        diameter = 100
        enclosable, circle = enclosable_in_circle(points, diameter/1000) 

        # Criteria for 'stop'
        stop = enclosable

        if not identical and not stop:
            c = calculate_course(trajectory) 
            c_ch = calculate_course_changes(c)
            c_c_ch = np.cumsum(c_ch) # Cummulative course changes

            # Criteria for 'fc' fixed-course
            if len(c) == 1:
                # The trajectory consists of only two vertices that are
                # not identical, so only one course can be determined. 
                fc = True
            else:
                # The course throughout the trajectory stays within a 
                # range determined by dc
                fc = (c.max() - c.min()) <= dc
            if not fc:
                # Criteria for 'tm' turn-manoeuver

                """
                When a ship changes its turning direction (e.g. 
                turning portside to turning starboard), some parts 
                of the trajectory may appear as 'fc' segments. This 
                definition of 'tm' allows for a limited number of 'fc'
                segments.  
                """
                tm = not contains_fc(c_ch,dc,3)
            else:
                tm = False
        else:
            fc = False
            tm = False
      
        types = ['stop','fc','tm']
        values = [stop,fc,tm]
        match = [i for i, x in enumerate(values) if x]
        
        if match:
            if len(match) > 1:
                raise Exception('Two types cannot be true! ',match)
            else:
                label = types[match[0]]

    if label == 'stop': 
        data = {
            'circle': circle,
        }
    elif label == 'fc' or label == 'tm':
        data = {
            'course': c,
            'course_changes': c_ch,
            'cum_course_changes': c_c_ch,
        }
    else:
        data = {}
            
    segment = {
        'type':label,
        'start_time': trajectory[0][2],
        'end_time': trajectory[-1][2],
        'start_loc':trajectory[0][0:2],
        'end_loc':trajectory[-1][0:2],
        'vertices':trajectory,
        'data': data,
        }

    if label:
        return  True, segment
    else:
        return False, segment

def calculate_course(trajectory):
    """
    Output: course along a trajectory as an numpy array.

    Input: trajectory as a list of (longitude, latitude, timestamp) tuples
    """
   
    TO_DEGREES = 180/np.pi
    TO_RADIANS = np.pi/180
    courses = []

    for i in range(1,len(trajectory)):

        pointB = trajectory[i][0:2]
        pointA = trajectory[i-1][0:2]

        diff_lat = pointB[1] - pointA[1]
        min_lat = min(pointB[1], pointA[1])
        diff_lon = pointB[0] - pointA[0]
        med_lat = diff_lat/2 + min_lat

        a = diff_lon * np.cos(med_lat * TO_RADIANS)
        if diff_lat > 0 and diff_lon > 0:
            course = 90 - (np.arctan(diff_lat / a) * TO_DEGREES)
        elif diff_lat > 0 and diff_lon < 0:
            course = 270 + (np.arctan(diff_lat / (-1 * a)) * TO_DEGREES)
        elif diff_lat < 0 and diff_lon > 0:
            course = 90 + (np.arctan(-1 * diff_lat / a) * TO_DEGREES)
        elif diff_lat < 0 and diff_lon < 0:
            course = 270 - (np.arctan(-1 * diff_lat / (-1 * a)) * TO_DEGREES)
        elif diff_lat < 0 and diff_lon == 0:
            course = 180
        elif diff_lat > 0 and diff_lon == 0:
            course = 0
        elif diff_lat == 0 and diff_lon > 0:
            course = 90
        elif diff_lat == 0 and diff_lon < 0:
            course = 270
        elif diff_lat == 0 and diff_lon == 0:
            course = np.nan

        courses.append(course)
        
    if courses:
        return np.array(courses)
    else:
        return np.array([])

def calculate_course_changes(course):
    d_c_v = []
    for d_c in np.diff(course):
        if abs(d_c) >  360 - abs(d_c):
            if d_c > 0:
                d_c_v.append(d_c - 360)
            else:
                d_c_v.append(360 + d_c)
        else:
            d_c_v.append(d_c)
    return np.array(d_c_v)

def contains_fc(c_ch,dc,k):
    r = (abs(c_ch) <= dc)
    for i in range(0,k-1):
        for j in range(0,len(r)):
            try: 
                r[j] = r[j] == True and r[j+1] == True
            except:
                r[j] = False 
    return (r).any() 

def reduce_waypoints(waypoints):
    # Remove waypoints if they mark a course change of less than 5 degrees. 
    waypoints = waypoints[:]
    i = 0
    while len(waypoints) >= 3 and i < len(waypoints) - 2:
        c = calculate_course([waypoints[i],waypoints[i+1],waypoints[i+2]])
        d_c = abs(c[1] - c[0])
        if d_c <= 5:
            waypoints.pop(i+1)
        else:
            i += 1
    return waypoints


def translate(segments,max_ctd=1500):

    subtrajectories = [] # [dict,dict,...]
    waypoints = [] # [(longitude,latitude,timestamp),...]

    # For translation, it is preferable to join all adjacent 'tm' segments
    segments = join_tm_segments(segments)
    
    for i in range(0,len(segments)):

        # Previous segment
        p_segment = segments[i-1] if i != 0 else None
        # Current segment
        c_segment = segments[i]
        # Next segment
        n_segment = segments[i+1] if i != len(segments) - 1 else None

        """
        Translate 'stop' segments (stop):
        """
        if c_segment['type'] == 'stop':
            if len(waypoints) != 0:
                waypoints.append(p_segment['vertices'][-1])
                subtrajectories.append({
                            'navigating':True,
                            'type':'waypoints',
                            'vertices':reduce_waypoints(waypoints),
                            })
            subtrajectories.append({
                'navigating':True,
                'type':'stop',
                'vertices':[(
                    c_segment['data']['circle'][0],
                    c_segment['data']['circle'][1],
                    c_segment['start_time']
                    ),(
                    c_segment['data']['circle'][0],
                    c_segment['data']['circle'][1],
                    c_segment['end_time']
                    )],
                })         
        """
        Translate 'fc' (fixed-course) segments
        """
        if c_segment['type'] == 'fc':
            if len(waypoints) == 0:
                waypoints.append(c_segment['vertices'][0])
            else:
                if p_segment['type'] == 'fc':
                    waypoints.append(c_segment['vertices'][0])
            if n_segment == None:
                waypoints.append(c_segment['vertices'][-1])
                subtrajectories.append({
                    'navigating':True,
                    'type':'waypoints',
                    'vertices':reduce_waypoints(waypoints),
                    })

        """
        Translate 'tm' (turn-manoeuver) segments 
        """
        if c_segment['type'] == 'tm':
            
            curvy = (c_segment['data']['cum_course_changes'] > 90).any()
            if p_segment == None or n_segment == None:
                bounded_by_fc_segments = False
            else:
                bounded_by_fc_segments = (p_segment['type'] == 'fc' and 
                n_segment['type'] == 'fc')
            
            if not curvy and bounded_by_fc_segments:
                """
                Try to translate 'tm' segment to a waypoint.
                The 'tm' segment is translatable if none of its vertices
                are farther than 500 meters from simplification.
                """
                point_a1 = p_segment['vertices'][0]
                point_a2 = p_segment['vertices'][-1]
                point_b1 = n_segment['vertices'][-1]
                point_b2 = n_segment['vertices'][0]
                translatable, ip, d = translate_tm_to_waypoint( \
                    c_segment['vertices'], \
                    point_a1=point_a1, point_a2=point_a2, \
                    point_b1=point_b1, point_b2=point_b2,
                    max_dist=max_ctd)
                if translatable:
                    avg_time = c_segment['start_time'] + \
                        (c_segment['end_time'] - c_segment['start_time'])/2
                    waypoints.append((ip[0],ip[1],avg_time))
                else:
                    if len(waypoints) != 0:
                        waypoints.append(p_segment['vertices'][-1])
                        subtrajectories.append({
                            'navigating':True,
                            'type':'waypoints',
                            'vertices':reduce_waypoints(waypoints),
                            })
                        waypoints = []
                    subtrajectories.append({
                        'navigating':False,
                        'type':None,
                        'vertices': c_segment['vertices'],
                        })
            else:
                if len(waypoints) != 0:
                    waypoints.append(p_segment['vertices'][-1])
                    subtrajectories.append({
                        'navigating':True,
                        'type':'waypoints',
                        'vertices':reduce_waypoints(waypoints),
                        })
                    waypoints = []
                subtrajectories.append({
                    'navigating':False,
                    'type':None,
                    'vertices': c_segment['vertices'],
                    })

            """
            Translate 'crumb' segments 
            """
            if c_segment['type'] == 'crumb':
                subtrajectories.append({
                    'navigating':False,
                    'type':None,
                    'vertices':c_segment['vertices'],
                    })    
    return subtrajectories


def join_tm_segments(segments):
    segments = segments[:]
    joined_segments = []
    while len(segments) >= 2: 
        fst = segments[0]
        sec = segments[1]
        if fst['type'] == 'tm' and sec['type'] == 'tm':
            segments.pop(0)
            fst['vertices'].extend(sec['vertices'])
            vertices = fst['vertices']
            segments[0] = {
                'type':'tm',
                'start_time': fst['start_time'],
                'end_time': sec['end_time'],
                'start_loc': fst['start_loc'],
                'end_loc': sec['end_loc'],
                'vertices': vertices,
                'data': {
                        'course':np.append(fst['data']['course'],sec['data']['course']),
                        'course_changes':np.append(fst['data']['course_changes'],sec['data']['course_changes']),
                        'cum_course_changes': np.append(fst['data']['cum_course_changes'], sec['data']['cum_course_changes']),
                    },
                }
            continue
        else:
            joined_segments.append(segments.pop(0))
            continue
    # Deal with the last segment in the list 'segment'
    joined_segments.append(segments.pop(0))
    return joined_segments

def translate_tm_to_waypoint(traj,max_dist=100,point_a1=None,point_a2=None,point_b1=None,point_b2=None):
    l_miss = False
    r_miss = False
    if point_a1 == None or point_a2 == None:
        point_a1 = traj[0]
        point_a2 = traj[1]
        l_miss = True
    if point_b1 == None or point_b2 == None:
        point_b1 = traj[-1]
        point_b2 = traj[-2]
        r_miss = True
    found, data = calculate_RL_intersection_point(point_a1,point_a2,point_b1,point_b2)
    if found:
        [ip, [old_left, new, old_right], theta] = data
        if theta >= 90:

            if l_miss and r_miss:
                ts = sample_within_segment(traj[1:-1])
            elif l_miss and not r_miss:
                ts = sample_within_segment(traj[1:])
            elif not l_miss and r_miss:
                ts = sample_within_segment(traj[:-1])
            else:
                ts = sample_within_segment(traj)

            if not old_left and not old_right:
                old = ts
                ctd = abs(calculate_cross_track_distance_RL(old[0],old[-1],ip))
            elif old_left and not old_right:
                old_left.extend(ts)
                old = old_left
                ctd = abs(calculate_cross_track_distance_RL(old[0],old[-1],old_left[-1]))*2
            elif not old_left and old_right:
                ts.extend(old_right)
                old = ts
                ctd = abs(calculate_cross_track_distance_RL(old[0],old[-1],old_right[0]))*2
            else:
                raise Exception('This shouldnt be an option!')

            d =check_fit([new[0],ip,new[-1]],old)
            
            if d < max_dist:
                return True, ip, d    
    
    return False, None, None

def calculate_RL_intersection_point(point_a1,point_a2,point_b1,point_b2):
    # Calculates the intersection point between two rhumb lines defined
    # by two points each. 

    # Transformt to Cartesian coordinates
    point_a1 = M(point_a1[0],point_a1[1])
    point_a2 = M(point_a2[0],point_a2[1])
    point_b1 = M(point_b1[0],point_b1[1])
    point_b2 = M(point_b2[0],point_b2[1])
    # Find intersection point
    found, data = line_intersection(point_a1,point_a2,point_b1,point_b2)
    if found:
        [ip,pnts,theta] = data
        for i in range(0,len(pnts)):
            if pnts[i]:
                x, y = zip(*pnts[i])
                # Transform to Earth coordinates
                lon, lat = M(x, y, inverse=True)
                pnts[i] = list(zip(lon,lat))
        ip_lon, ip_lat = M(ip[0], ip[1], inverse=True)
        return True, [[ip_lon,ip_lat],pnts,theta]
    else:
        return False, []

def sample_within_segment(traj):
    points = []
    for i in range(1,len(traj)):
        point_b = traj[i]
        point_a = traj[i-1]
        x0, y0 = M(point_a[0],point_a[1])
        x1, y1 = M(point_b[0],point_b[1])
        p0 = np.array([x0,y0])
        p1 = np.array([x1,y1])
        l = lambda s: p0 + s*(p1 - p0)
        xy = list(map(lambda s: p0 + s*(p1 - p0), np.linspace(0,1,5)))
        lonlat = list(map(lambda xy: M(xy[0], xy[1], inverse=True),xy))
        points.extend(lonlat)
        if i != len(traj)-1:
            points.pop(-1)
    return points

def check_fit(simple,complex):
    # Calculates the maximum cross-track distance from all the points in complex
    # to the simple trajectory.
    d = np.ones(len(complex))*np.inf
    for i in range(1,len(simple)):
        for j in range(0,len(complex)):
            ctd = calculate_cross_track_distance_RL(simple[i-1],simple[i],complex[j])
            if not np.isnan(ctd):
                d[j] = min(ctd,d[j])
    return max(d)

def line_intersection(point_a1,point_a2,point_b1,point_b2):
    """
    Determines whether the intersection between lines passing through
    points a an b exists. If existing,
    returns the intersection point, two arrays containing sample points from 
    the start of the rays to the intersection point, and the angle between the
    rays. If not existing, returns False.

    Based on the algorithms presented at:
    http://geomalgorithms.com/a05-_intersect-1.html#intersect2D_2Segments()
    """
    p0 = np.array(point_a1)
    p1 = np.array(point_a2)
    q0 = np.array(point_b1)
    q1 = np.array(point_b2)
    u = p1 - p0
    v = q1 - q0
    w = p0 - q0
    d = np.cross(u,v)
    # ... The lines are parallel
    if abs(d) < 0.000001:
        return False, []
    else:
        s = np.cross(v,w)/d
        t = np.cross(u,w)/d
        # ... Invalid intersection
        if s < 1 and t < 1: 
            return False, []
        else:
            old_left = []
            old_right = []
            if s > 1 and t > 1:
                new_a = list(map(lambda s: p0 + s*u,np.linspace(1,s,21)))
                new_b = list(map(lambda t: q0 + t*v,np.linspace(t,1,20)))
                ip = new_a.pop(-1)
                new_a.extend(new_b)
                new = new_a 
                theta = abs(np.arctan2(np.cross((p1 - ip),(q1 - ip)),np.dot((p1 - ip),(q1 - ip)))*TO_DEGREES)
            elif 0 < s < 1 and t > 1:
                new = list(map(lambda t: q0 + t*v,np.linspace(t,1,20)))
                old_left = list(map(lambda s: p0 + s*u,np.linspace(s,1,20)))
                ip = new[0]
                theta = 180 - abs(np.arctan2(np.cross((p1 - ip),(q1 - ip)),np.dot((p1 - ip),(q1 - ip)))*TO_DEGREES)
            elif s > 1 and 0 < t < 1:
                new = list(map(lambda s: p0 + s*u,np.linspace(1,s,20)))
                old_right = list(map(lambda t: q0 + t*v,np.linspace(1,t,20)))
                ip = new[-1]
                theta = 180 - abs(np.arctan2(np.cross((p1 - ip),(q1 - ip)),np.dot((p1 - ip),(q1 - ip)))*TO_DEGREES)
            else:
                return False, []
            return True, [tuple(ip),[old_left,new,old_right],theta]  

def calculate_cross_track_distance_RL(point_a1,point_a2,point_b):
    # Cross track distance of point_b from rhumb line defined by 
    # point_a1 and point_a2
    point_a1_xy = M(point_a1[0],point_a1[1])
    point_a2_xy = M(point_a2[0],point_a2[1])
    point_b_xy = M(point_b[0],point_b[1])
    alpha = np.arctan2(point_a2_xy[1]-point_a1_xy[1],point_a2_xy[0]-point_a1_xy[0]) + np.pi/2
    found, ip_xy = find_ray_segment_intersection(point_a1_xy,point_a2_xy,point_b_xy,alpha)
    if found:
        ip = M(ip_xy[0],ip_xy[1],inverse=True)
        d = great_circle(point_b,ip)
        return d
    else:
        d1 = great_circle(point_b,point_a1)
        d2 = great_circle(point_b,point_a2)
        return min(d1,d2)

def find_ray_segment_intersection(point_a1,point_a2,point_b,alpha):
    p0 = np.array(point_a1)
    p1 = np.array(point_a2)
    u = p1 -p0
    q0 = np.array(point_b)
    v = np.array([np.cos(alpha), np.sin(alpha)])
    d = np.cross(u,v)
    w = p0 - q0
    # ... The segment and ray are parallel
    if abs(d) < 0.000001:
        return False, None
    else:
        s = round(np.cross(v,w)/d,15)   
        # ... The ray intersects the segment
        if 0 <= s <= 1:
            ip = p0 + s*u
            return True, ip
        # ... The ray does not intersect the segment
        else:
            return False, None

def great_circle(point_a,point_b):
    point_a = (point_a[1],point_a[0])
    point_b = (point_b[1],point_b[0])
    point_a = nv.GeoPoint(point_a[0],point_a[1],degrees=True)
    point_b = nv.GeoPoint(point_b[0],point_b[1],degrees=True)
    d, _a1, _a2 = point_a.distance_and_azimuth(point_b)
    return d


def plot_segments(segments,mercator=False):
    if mercator:
        first = True
        for s in segments: 
            lon, lat, t = zip(*s['vertices'])
            if first:
                l_l_c_lat = min(lat)
                l_l_c_lon = min(lon)
                u_r_c_lat = max(lat)
                u_r_c_lon = max(lon)
                first = False
            else:
                l_l_c_lat = min([l_l_c_lat,min(lat)])
                l_l_c_lon = min([l_l_c_lon,min(lon)])
                u_r_c_lat = max([u_r_c_lat,max(lat)])
                u_r_c_lon = max([u_r_c_lon,max(lon)])
        m = Basemap(projection='merc', \
            llcrnrlat=l_l_c_lat-0.05, \
            llcrnrlon=l_l_c_lon-0.05, \
            urcrnrlat=u_r_c_lat+0.05, \
            urcrnrlon=u_r_c_lon+0.05, \
            resolution='h')
        m.drawcoastlines()

    for s in segments:
        if s['type'] == 'fc':
            clr = 'c'
        elif s['type'] == 'tm':
            clr = 'm'
        elif s['type'] == 'crumb':
            clr = 'k'
        else:
            clr = 'r'
        if s['type'] == 'stop':
            c = s['data']['circle']
            x, y, d = c
            if mercator:
                x, y = m(x,y)
            plt.plot(x,y, marker='o', markersize=12, color="r", alpha=0.3)
        if mercator:
            lon, lat, t = zip(*s['vertices'])
            x, y = m(lon,lat)
        else:
            x, y, t = zip(*s['vertices'])
        plt.plot(x,y,marker='.',color=clr,alpha=0.5)
        plt.axis('off')

def plot_subtrajectories(subtrajectories,trajectory=None,mercator=False):
    
    if mercator:
        first = True
        for st in subtrajectories: 
            lon, lat, t = zip(*st['vertices'])
            if first:
                l_l_c_lat = min(lat)
                l_l_c_lon = min(lon)
                u_r_c_lat = max(lat)
                u_r_c_lon = max(lon)
                first = False
            else:
                l_l_c_lat = min([l_l_c_lat,min(lat)])
                l_l_c_lon = min([l_l_c_lon,min(lon)])
                u_r_c_lat = max([u_r_c_lat,max(lat)])
                u_r_c_lon = max([u_r_c_lon,max(lon)])
        m = Basemap(projection='merc', \
            llcrnrlat=l_l_c_lat-0.05, \
            llcrnrlon=l_l_c_lon-0.05, \
            urcrnrlat=u_r_c_lat+0.05, \
            urcrnrlon=u_r_c_lon+0.05, \
            resolution='h')
        m.drawcoastlines()

    vertices = []
    for st in subtrajectories:

        if mercator:
            lon, lat, t = zip(*st['vertices'])
            x, y = m(lon,lat)
        else:
            x, y, t = zip(*st['vertices'])  
        vertices.extend(st['vertices'])
        if st['navigating']:
            if st['type'] == 'stop':
                clr = 'r'
            else:
                clr = 'c'
        else:
            clr = 'm'
        plt.plot(x,y,marker='.',color=clr,alpha=0.7,markersize=8)
        if st['type'] == 'stop':
            plt.plot(x[-1],y[-1],marker='o',color=clr,alpha=0.7,markersize=12)
      
    plt.axis('off')
    txt = 'n_vertices = %d' % len(vertices)
    axes = plt.gca()
    ymin, ymax = axes.get_ylim()
    xmin, xmax = axes.get_xlim()
  
    if trajectory != None:
        lon, lat, t = zip(*trajectory)
        x,y = m(lon,lat)
        plt.plot(x,y,color='grey',alpha=0.9)

def plot_trajectory(o_trajectory,color='g',trajectory=None,mercator=False):

    if mercator:
           
        lon, lat, t = zip(*o_trajectory)
        l_l_c_lat = min(lat)
        l_l_c_lon = min(lon)
        u_r_c_lat = max(lat)
        u_r_c_lon = max(lon)         
        m = Basemap(projection='merc', \
            llcrnrlat=l_l_c_lat-0.05, \
            llcrnrlon=l_l_c_lon-0.05, \
            urcrnrlat=u_r_c_lat+0.05, \
            urcrnrlon=u_r_c_lon+0.05, \
            resolution='h')
        m.drawcoastlines()

    if len(o_trajectory) != 0:
        if mercator:
            lon, lat, t = zip(*o_trajectory)
            x, y = m(lon,lat)
        else:
            x, y, t = zip(*o_trajectory)
        plt.plot(x,y,marker='.',color='grey',markersize=5)
    
    txt = 'n_vertices = %d' % len(o_trajectory)
    plt.axis('off')
    axes = plt.gca()
    ymin, ymax = axes.get_ylim()
    xmin, xmax = axes.get_xlim()
  
    if trajectory != None:
        lon, lat, t = zip(*trajectory)
        x,y = m(lon,lat)
        plt.plot(x,y,color='grey',alpha=0.9)
      
