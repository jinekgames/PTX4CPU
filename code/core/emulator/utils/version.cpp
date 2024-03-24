#include <version.h>


#ifndef PROJECT_GIT_COMMIT
#define PROJECT_GIT_COMMIT "NO_COMMIT_INFORMATION";
#endif

extern "C" const char ProjectGitCommit[41] = PROJECT_GIT_COMMIT;