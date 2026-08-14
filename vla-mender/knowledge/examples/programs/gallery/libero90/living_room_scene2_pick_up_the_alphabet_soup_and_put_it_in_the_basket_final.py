"""
LIBERO_90 LIVING_ROOM_SCENE2: pick up the alphabet soup and put it in the basket

Scene analysis (seeds 51-53):
- Alphabet soup (yellow can): X~0.46, Y~-0.14 to -0.17, Z~0.075 (negative Y = left side)
- Tomato sauce (red can): X~0.42-0.44, Y~0.04-0.06, Z~0.075 (positive Y = right side)
- Pudding/cream cheese (flat box): Z~0.035 (much lower)
- Basket: X~0.58, Y~0.26, top_z~0.16

Strategy:
1. Detect alphabet soup: prompt "alphabet soup yellow can" + filter Y<-0.05 + Z in [0.05, 0.10]
2. Detect basket: prompt "basket" or "wicker basket"
3. Top-down grasp at can centroid + GraspNet refinement
4. Lift, transport to basket, drop in basket
"""

import numpy as np

TOP_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])


def localize_alphabet_soup(rgb, depth_img, K, E):
    """Find the yellow alphabet soup can. Filter by Y<-0.05 and Z in [0.05, 0.10]."""
    prompts = [
        "alphabet soup yellow can",
        "yellow soup can",
        "alphabet soup",
        "alphabet soup can",
    ]
    best_score = -1.0
    best_mask = None
    best_pts = None
    best_prompt = None
    for prompt in prompts:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        # Top 8 by score, filter by 3D
        for m in sorted(masks, key=lambda d: d["score"], reverse=True)[:8]:
            mask = m["mask"].astype(np.uint8)
            pts = mask_to_world_points(mask, depth_img, K, E)
            if pts is None or len(pts) < 30:
                continue
            ctr = np.median(pts, axis=0)
            z_min, z_max = pts[:, 2].min(), pts[:, 2].max()
            # Filter: yellow can on LEFT side, at can height
            if ctr[1] > -0.05:  # must be at negative Y (left)
                continue
            if not (0.04 <= ctr[2] <= 0.10):  # cylindrical can height range
                continue
            if z_max - z_min < 0.04:  # not flat — needs vertical extent
                continue
            if z_max - z_min > 0.20:  # not too tall — exclude scene clutter
                continue
            # Within 3D bbox of expected can
            if ctr[0] < 0.30 or ctr[0] > 0.55:
                continue
            print(f"   [soup] '{prompt}' score={m['score']:.3f}, 3D=({ctr[0]:.3f},{ctr[1]:.3f},{ctr[2]:.3f})", flush=True)
            if m["score"] > best_score:
                best_score = m["score"]
                best_mask = mask
                best_pts = pts
                best_prompt = prompt
        if best_mask is not None and best_score > 0.15:
            break
    return best_mask, best_pts, best_prompt


def localize_basket(rgb, depth_img, K, E):
    """Find the basket. Use p10/p90 of the basket-rim points for true center.
    Filter to elevated basket points (Z > 0.05) to focus on rim/walls, not the floor reflection.
    """
    prompts = ["basket", "wicker basket", "wire basket", "woven basket"]
    for prompt in prompts:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        best = max(masks, key=lambda d: d["score"])
        mask = best["mask"].astype(np.uint8)
        pts = mask_to_world_points(mask, depth_img, K, E)
        if pts is None or len(pts) < 100:
            continue
        # Filter to elevated basket points (Z > 0.05 = above table) and Y > 0.15 (basket on right)
        pts_f = pts[(pts[:, 2] > 0.05) & (pts[:, 1] > 0.15) & (pts[:, 0] > 0.40)]
        if len(pts_f) < 50:
            pts_f = pts[(pts[:, 2] > 0.02) & (pts[:, 1] > 0.10)]
        # p10/p90 midpoint for true center (handles asymmetric SAM3 mask coverage)
        bx = (np.percentile(pts_f[:, 0], 10) + np.percentile(pts_f[:, 0], 90)) / 2
        by = (np.percentile(pts_f[:, 1], 10) + np.percentile(pts_f[:, 1], 90)) / 2
        top_z = np.percentile(pts_f[:, 2], 90)
        center = np.array([bx, by, np.median(pts_f[:, 2])])
        print(f"   [basket] '{prompt}' score={best['score']:.3f}, center=({bx:.3f},{by:.3f},{np.median(pts_f[:,2]):.3f}), top_z={top_z:.3f}, n_pts={len(pts_f)}", flush=True)
        # Also report extents
        print(f"   [basket] X=[{pts_f[:,0].min():.3f},{pts_f[:,0].max():.3f}], Y=[{pts_f[:,1].min():.3f},{pts_f[:,1].max():.3f}], Z=[{pts_f[:,2].min():.3f},{pts_f[:,2].max():.3f}]", flush=True)
        return mask, pts_f, center, top_z
    return None, None, None, None


def run():
    goto_home_joint_position()

    obs = get_observation()
    cam = obs["agentview"]
    rgb = cam["images"]["rgb"]
    depth = cam["images"]["depth"]
    K = cam["intrinsics"]
    E = cam["pose_mat"]
    depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth

    # === Localize alphabet soup ===
    print("\n[1] Localize alphabet soup", flush=True)
    soup_mask, soup_pts, soup_prompt = localize_alphabet_soup(rgb, depth_img, K, E)
    if soup_mask is None:
        print("ERROR: alphabet soup not found", flush=True)
        return False
    soup_center = np.median(soup_pts, axis=0)
    soup_z_min = soup_pts[:, 2].min()
    soup_z_max = soup_pts[:, 2].max()
    print(f"   center={soup_center.round(3)}, Z=[{soup_z_min:.3f},{soup_z_max:.3f}]", flush=True)

    # === Localize basket (do this BEFORE grasping; arm may occlude post-lift) ===
    print("\n[2] Localize basket", flush=True)
    basket_mask, basket_pts, basket_center, basket_top_z = localize_basket(rgb, depth_img, K, E)
    if basket_mask is None:
        print("ERROR: basket not found", flush=True)
        return False

    # === Grasp position: use 3D bbox center (cylindrical, symmetric) ===
    print("\n[3] Compute grasp pose (bbox center for cylinder)", flush=True)
    cx = (np.percentile(soup_pts[:, 0], 10) + np.percentile(soup_pts[:, 0], 90)) / 2
    cy = (np.percentile(soup_pts[:, 1], 10) + np.percentile(soup_pts[:, 1], 90)) / 2
    grasp_pos = np.array([cx, cy, soup_center[2] + 0.02])
    print(f"   bbox-center grasp_pos={grasp_pos.round(3)}, median={soup_center.round(3)}", flush=True)

    # === Execute grasp ===
    print("\n[4] Execute grasp", flush=True)
    open_gripper()
    hover = grasp_pos.copy()
    hover[2] += 0.15
    goto_pose(hover, TOP_DOWN_QUAT)
    goto_pose(grasp_pos, TOP_DOWN_QUAT)
    close_gripper()

    # === Verify grasp ===
    obs2 = get_observation()
    gw = obs2["robot_cartesian_pos"][-1]
    ee_xyz = obs2["robot_cartesian_pos"][:3]
    print(f"   gw after close = {gw:.4f}, ee_xyz={ee_xyz.round(3)}", flush=True)
    # Threshold: gw is normalized (0=closed, 1=open). For a 6.5cm soup can, gw~0.7
    # gw < 0.10 means gripper closed mostly = nothing held / very thin contact
    if gw < 0.10:
        for retry_dz in [-0.015, -0.030, +0.010]:
            print(f"   AIR GRASP (gw={gw:.4f}), retry with dz={retry_dz}", flush=True)
            open_gripper()
            retry = grasp_pos.copy()
            retry[2] += retry_dz
            goto_pose(hover, TOP_DOWN_QUAT)
            goto_pose(retry, TOP_DOWN_QUAT)
            close_gripper()
            obs2 = get_observation()
            gw = obs2["robot_cartesian_pos"][-1]
            ee_xyz = obs2["robot_cartesian_pos"][:3]
            print(f"   retry gw = {gw:.4f}, ee_xyz={ee_xyz.round(3)}", flush=True)
            if gw >= 0.10:
                break
        if gw < 0.10:
            print("   STILL air grasp, abort", flush=True)
            open_gripper()
            goto_home_joint_position()
            return False

    # === Lift incrementally (avoid IK discontinuity) ===
    print("\n[5] Lift incrementally", flush=True)
    # Get current ee z (after grasp it's clamped at ~0.21 for low-z grasp targets)
    obs_g = get_observation()
    cur_z = obs_g["robot_cartesian_pos"][2]
    # Step from cur_z up to 0.27 in 0.02 increments (smooth IK transitions)
    target_z = 0.27
    if cur_z < target_z:
        z_steps = list(np.arange(cur_z + 0.02, target_z + 0.001, 0.02))
        for zs in z_steps:
            lift = np.array([grasp_pos[0], grasp_pos[1], zs])
            goto_pose(lift, TOP_DOWN_QUAT)
    obs_lift = get_observation()
    gw_lift = obs_lift["robot_cartesian_pos"][-1]
    ee_lift = obs_lift["robot_cartesian_pos"][:3]
    print(f"   after lift: gw={gw_lift:.4f}, ee_xyz={ee_lift.round(3)}", flush=True)

    # === Transport to basket ===
    print("\n[6] Transport to above basket", flush=True)
    above = np.array([basket_center[0], basket_center[1], 0.27])
    goto_pose(above, TOP_DOWN_QUAT)

    # === Drop into basket (incremental descent) ===
    place_z = basket_top_z + 0.10  # 10cm above rim
    print(f"\n[7] Incremental descent into basket (basket_top_z={basket_top_z:.3f}, place_z={place_z:.3f})", flush=True)
    for z in [0.25, 0.23, place_z]:
        wpt = np.array([basket_center[0], basket_center[1], z])
        goto_pose(wpt, TOP_DOWN_QUAT)
    open_gripper()

    # Settle
    for _ in range(10):
        get_observation()

    goto_home_joint_position()
    return True


success = run()
print(f"\nRESULT: success={success}", flush=True)
