from .base_api import list_apis, register_api
from .gym_action import GymActionApi

register_api("GymActionApi", GymActionApi)

try:
    from .franka.control import FrankaControlApi
    from .franka.control_privileged import FrankaControlPrivilegedApi
    from .franka.control_reduced import FrankaControlApiReduced
    from .franka.control_reduced_skill_library import FrankaControlApiReducedSkillLibrary
    from .franka.control_reduced_exampleless import FrankaControlApiReducedExampleless
    from .franka.nut_assembly_privileged import FrankaControlNutAssemblyPrivilegedApi
    from .franka.nut_assembly_visual import FrankaControlNutAssemblyVisualApi
    from .franka.spill_wipe import FrankaControlSpillWipeApi
    from .franka.spill_wipe_privileged import FrankaControlSpillWipePrivilegedApi
    from .franka.handover_privileged import FrankaHandoverPrivilegedApi
    from .franka.handover import FrankaHandoverApi
    from .franka.handover_reduced import FrankaHandoverApiReduced
    from .franka.handover_reduced_exampleless import FrankaHandoverApiReducedExampleless
    from .franka.two_arm_lift import FrankaTwoArmLiftApi
    from .franka.two_arm_lift_privileged import FrankaTwoArmLiftPrivilegedApi

    register_api("FrankaControlPrivilegedApi", FrankaControlPrivilegedApi)
    register_api("FrankaControlApi", lambda env: FrankaControlApi(env, use_sam3=True))
    register_api("FrankaControlApiReduced", FrankaControlApiReduced)
    register_api(
        "FrankaControlApiReducedBimanual", lambda env: FrankaControlApiReduced(env, bimanual=True)
    )
    register_api(
        "FrankaControlApiReducedExamplelessBimanual",
        lambda env: FrankaControlApiReducedExampleless(env, bimanual=True),
    )
    register_api(
        "FrankaControlApiReducedBimanualHandover",
        lambda env: FrankaControlApiReduced(env, bimanual=True, is_handover=True),
    )
    register_api(
        "FrankaControlApiReducedExamplelessBimanualHandover",
        lambda env: FrankaControlApiReducedExampleless(env, bimanual=True, is_handover=True),
    )
    register_api(
        "FrankaControlApiReducedSpillWipe",
        lambda env: FrankaControlApiReduced(env, tcp_offset=[0.0, 0.0, -0.0158]),
    )
    register_api("FrankaControlApiReducedExampleless", FrankaControlApiReducedExampleless)

    register_api("FrankaControlApiReducedSkillLibrary", FrankaControlApiReducedSkillLibrary)
    register_api(
        "FrankaControlApiReducedSkillLibraryBimanual",
        lambda env: FrankaControlApiReducedSkillLibrary(env, bimanual=True),
    )
    register_api(
        "FrankaControlApiReducedSkillLibrarySpillWipe",
        lambda env: FrankaControlApiReducedSkillLibrary(env, tcp_offset=[0.0, 0.0, -0.0158]),
    )
    register_api(
        "FrankaControlApiReducedSkillLibraryBimanualHandover",
        lambda env: FrankaControlApiReducedSkillLibrary(env, bimanual=True, is_handover=True),
    )

    register_api(
        "FrankaControlSpillWipeApi",
        lambda env: FrankaControlSpillWipeApi(env, tcp_offset=[0.0, 0.0, -0.0158], use_sam3=True),
    )
    register_api(
        "FrankaControlSpillWipeApiReduced",
        lambda env: FrankaControlApiReduced(
            env, tcp_offset=[0.0, 0.0, -0.0158], is_spill_wipe=True
        ),
    )
    register_api(
        "FrankaControlSpillWipePrivilegedApi",
        lambda env: FrankaControlSpillWipePrivilegedApi(env, tcp_offset=[0.0, 0.0, -0.0158]),
    )
    register_api(
        "FrankaControlSpillWipeApiReducedExampleless",
        lambda env: FrankaControlApiReducedExampleless(
            env, tcp_offset=[0.0, 0.0, -0.0158], is_spill_wipe=True
        ),
    )

    register_api("FrankaHandoverPrivilegedApi", FrankaHandoverPrivilegedApi)
    register_api("FrankaHandoverApi", FrankaHandoverApi)
    register_api("FrankaHandoverApiReduced", FrankaHandoverApiReduced)
    register_api("FrankaHandoverApiReducedExampleless", FrankaHandoverApiReducedExampleless)

    register_api("FrankaTwoArmLiftApi", FrankaTwoArmLiftApi)
    register_api("FrankaTwoArmLiftPrivilegedApi", FrankaTwoArmLiftPrivilegedApi)
    register_api(
        "FrankaTwoArmLiftApiReduced",
        lambda env: FrankaControlApiReduced(env, bimanual=True, use_sam3=False),
    )
    register_api(
        "FrankaTwoArmLiftApiReducedExampleless",
        lambda env: FrankaControlApiReducedExampleless(env, bimanual=True, use_sam3=False),
    )

    register_api("FrankaControlNutAssemblyPrivilegedApi", FrankaControlNutAssemblyPrivilegedApi)
    register_api("FrankaControlNutAssemblyVisualApi", FrankaControlNutAssemblyVisualApi)
    register_api(
        "FrankaControlNutAssemblyApiReduced",
        lambda env: FrankaControlApiReduced(env, is_peg_assembly=True),
    )
    register_api(
        "FrankaControlNutAssemblyApiReducedExampleless",
        lambda env: FrankaControlApiReducedExampleless(env, is_peg_assembly=True),
    )
    register_api(
        "FrankaControlMultiPrivilegedApi",
        lambda env: FrankaControlPrivilegedApi(env, multi_turn=True),
    )

    register_api(
        "FrankaRealReducedSkillLibraryControlApi",
        lambda env: FrankaControlApiReducedSkillLibrary(
            env, tcp_offset=[0.0, 0.0, -0.157], real=True
        ),
    )
    register_api(
        "FrankaRealControlApi",
        lambda env: FrankaControlApi(env, tcp_offset=[0.0, 0.0, -0.157], real=True),
    )

    try:
        from .franka.libero import FrankaLiberoApi
        from .franka.libero_reduced import FrankaLiberoApiReduced
        from .franka.libero_osc_reduced_skill_library import (
            FrankaLiberoApiReducedOscSkillLibrary,
        )
        from .franka.libero_reduced_skill_library import FrankaLiberoApiReducedSkillLibrary

        register_api("FrankaLiberoApi", lambda env: FrankaLiberoApi(env, use_sam3=True))
        register_api("FrankaLiberoApiReduced", FrankaLiberoApiReduced)
        register_api("FrankaLiberoApiReducedSkillLibrary", FrankaLiberoApiReducedSkillLibrary)
        register_api(
            "FrankaLiberoApiReducedOscSkillLibrary",
            FrankaLiberoApiReducedOscSkillLibrary,
        )
    except Exception:
        print("LIBERO not installed, skipping LIBERO APIs")
except Exception:
    print("Franka APIs not installed, skipping Franka APIs")

# LIBERO uses an older robosuite controller stack than the standalone Franka
# environments. Register its APIs independently so an optional standalone
# robosuite import cannot disable the LIBERO-only runtime.
try:
    from .franka.libero import FrankaLiberoApi
    from .franka.libero_reduced import FrankaLiberoApiReduced
    from .franka.libero_osc_reduced_skill_library import (
        FrankaLiberoApiReducedOscSkillLibrary,
    )
    from .franka.libero_reduced_skill_library import FrankaLiberoApiReducedSkillLibrary

    register_api("FrankaLiberoApi", lambda env: FrankaLiberoApi(env, use_sam3=True))
    register_api("FrankaLiberoApiReduced", FrankaLiberoApiReduced)
    register_api("FrankaLiberoApiReducedSkillLibrary", FrankaLiberoApiReducedSkillLibrary)
    register_api(
        "FrankaLiberoApiReducedOscSkillLibrary",
        FrankaLiberoApiReducedOscSkillLibrary,
    )
except Exception as exc:
    print(f"LIBERO APIs unavailable: {exc}")

try:
    from .r1pro.control import R1ProControlApi

    register_api("R1ProControlApi", lambda env: R1ProControlApi(env, use_sam3=True))
except Exception:
    print("R1Pro not installed, skipping R1Pro APIs")
