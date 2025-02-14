#pragma once


inline constexpr auto kProductVersion =
#ifdef PTX4CPU_VERSION
    PTX4CPU_VERSION;
#else  // PTX4CPU_VERSION
    "NO_VERSION_INFORMATION";
#pragma message("ERROR: Failed to get product version");
#endif  // PTX4CPU_VERSION

inline constexpr auto kProjectGitCommit =
#ifdef PROJECT_GIT_COMMIT
    PROJECT_GIT_COMMIT;
#else  // PROJECT_GIT_COMMIT
    "NO_COMMIT_INFORMATION";
#pragma message("ERROR: Failed to get Git repesitory commit");
#endif  // PROJECT_GIT_COMMIT
