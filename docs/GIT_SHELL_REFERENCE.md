# Git & Shell Commands Reference

**Purpose:** Complete reference of all Git and shell commands used in this project setup, with debugging techniques and explanations.

**Author:** Sudhir Shivaram
**Date:** December 2024
**Project:** Multimodal RAG Research & Finance

---

## Table of Contents

1. [Project Setup Commands](#project-setup-commands)
2. [Git Initialization](#git-initialization)
3. [Git Configuration](#git-configuration)
4. [GitHub Remote Setup](#github-remote-setup)
5. [Troubleshooting & Debugging](#troubleshooting--debugging)
6. [Common Git Issues & Solutions](#common-git-issues--solutions)
7. [Best Practices](#best-practices)
8. [Quick Reference](#quick-reference)

---

## Project Setup Commands

### Creating Directory Structure

```bash
# Create project root
mkdir -p ~/portfolio-project/multimodal-rag-research-finance

# Navigate to project
cd ~/portfolio-project/multimodal-rag-research-finance

# Create microservices directory structure
mkdir -p src/{services/{vision,search,llm,cache},routers,models,core,utils}
mkdir -p airflow/dags
mkdir -p tests/{unit,integration}
mkdir -p docs deployment scripts

# Verify structure
tree -L 3 -d
```

**Why:**
- `mkdir -p`: Creates parent directories if they don't exist (no error if already exists)
- `{a,b,c}`: Bash brace expansion - creates multiple directories in one command
- `tree`: Visual representation of directory structure

### Copying Files from Old Project

```bash
# Copy planning document
cp ~/arxiv-paper-curator/MULTIMODAL_RAG_PLAN.md ./docs/

# List copied files with details
ls -lh docs/
```

**Flags explained:**
- `-l`: Long format (permissions, owner, size, date)
- `-h`: Human-readable sizes (KB, MB instead of bytes)

---

## Git Initialization

### Initialize Repository

```bash
# Initialize git repository
git init

# Output: "Initialized empty Git repository in /path/to/project/.git/"
```

**What happens:**
- Creates `.git/` directory (hidden)
- Sets up Git tracking structure
- Creates default branch (usually `master`)

### Rename Branch to Main

```bash
# Rename branch from master to main
git branch -m main

# Verify current branch
git branch
```

**Why rename to main:**
- Modern Git convention (GitHub default)
- Inclusive terminology
- Industry standard

**Flag:**
- `-m`: Move/rename branch

### Check Repository Status

```bash
# See current status
git status
```

**Output shows:**
- Current branch
- Untracked files (new files not yet added)
- Staged files (ready to commit)
- Modified files
- Deleted files

**Color coding:**
- Red: Untracked or modified (not staged)
- Green: Staged (ready to commit)

---

## Git Configuration

### Set User Information

```bash
# Set user name (for commits)
git config user.name "Sudhir Shivaram"

# Set user email
git config user.email "Shivaram.Sudhir@gmail.com"

# Verify configuration
git config --list | grep user
```

**Scope levels:**
- `git config` (no flag): Repository-specific (current project only)
- `git config --global`: User-wide (all your repositories)
- `git config --system`: System-wide (all users)

**Best practice:**
- Use `--global` for personal projects
- Use repository-specific for work projects (different email)

### View All Configuration

```bash
# List all git config settings
git config --list

# List only user settings
git config --list | grep user

# Get specific setting
git config user.name
```

---

## Staging and Committing

### Add Files to Staging

```bash
# Add all files in current directory
git add .

# Add specific file
git add README.md

# Add specific directory
git add docs/

# Add all files matching pattern
git add *.md
```

**Staging area:**
- Intermediate area between working directory and repository
- Allows selective commits
- Can review changes before committing

### Create Commit

```bash
# Commit with inline message
git commit -m "Initial project setup: microservices architecture and documentation"

# Commit with editor (for longer messages)
git commit

# Amend last commit (fix message or add forgotten files)
git commit --amend
```

**Good commit messages:**
- Present tense ("Add feature" not "Added feature")
- Imperative mood ("Fix bug" not "Fixed bug")
- Concise but descriptive
- NO emojis (professional projects)
- NO co-author attribution (if not desired)

**Commit message structure:**
```
<type>: <short summary> (50 chars max)

<detailed description> (optional, 72 chars per line)

- Bullet points for changes
- Why this change was made
- Any breaking changes
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

### View Commit History

```bash
# Full commit log
git log

# One-line per commit
git log --oneline

# Show last 5 commits
git log -5

# Show commits with file changes
git log --stat

# Show commits by author
git log --author="Sudhir"

# Show commits in date range
git log --since="2024-12-01" --until="2024-12-31"
```

---

## GitHub Remote Setup

### Add Remote Repository

```bash
# Add remote named 'origin'
git remote add origin git@github.com:sushiva/multimodal-rag-research-finance.git

# Verify remotes
git remote -v
```

**Output:**
```
origin  git@github.com:sushiva/multimodal-rag-research-finance.git (fetch)
origin  git@github.com:sushiva/multimodal-rag-research-finance.git (push)
```

**Remote naming conventions:**
- `origin`: Primary remote (default)
- `upstream`: Original repository (for forks)
- `production`: Production deployment
- Custom names: `sushiva`, `sudhirshivaram`

### SSH vs HTTPS URLs

```bash
# SSH format (recommended)
git@github.com:username/repo.git

# HTTPS format
https://github.com/username/repo.git
```

**SSH benefits:**
- No password prompts
- More secure (key-based)
- Faster

**HTTPS benefits:**
- Works through firewalls
- No SSH key setup needed

### Test SSH Connection

```bash
# Test GitHub SSH authentication
ssh -T git@github.com

# Expected output:
# "Hi username! You've successfully authenticated, but GitHub does not provide shell access."
```

**If SSH fails:**
```bash
# Generate new SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"

# Start SSH agent
eval "$(ssh-agent -s)"

# Add key to agent
ssh-add ~/.ssh/id_ed25519

# Copy public key to clipboard
cat ~/.ssh/id_ed25519.pub
# Then add to GitHub Settings → SSH Keys
```

### Manage Remotes

```bash
# List remotes
git remote -v

# Add new remote
git remote add <name> <url>

# Remove remote
git remote remove <name>

# Rename remote
git remote rename <old-name> <new-name>

# Change remote URL
git remote set-url origin <new-url>
```

**Example: Fix typo in URL**
```bash
# Wrong URL (missing 't')
origin  git@github.com:sushiva/multimodal-rag-research-finance.gi

# Remove and re-add
git remote remove origin
git remote add origin git@github.com:sushiva/multimodal-rag-research-finance.git
```

---

## Pushing to GitHub

### Basic Push

```bash
# Push to remote 'origin', branch 'main'
git push origin main

# Set upstream and push (first time)
git push -u origin main
# OR
git push --set-upstream origin main

# After upstream is set, just:
git push
```

**Flags:**
- `-u` or `--set-upstream`: Links local branch to remote branch
- After setting upstream, `git push` knows where to push

### Force Push (USE WITH CAUTION)

```bash
# Force push (overwrites remote)
git push --force origin main

# Safer force push (only if no one else pushed)
git push --force-with-lease origin main
```

**WARNING:**
- Only use on your own branches
- NEVER force push to shared branches (main, develop)
- Can lose others' work

---

## Pulling from GitHub

### Fetch vs Pull

```bash
# Fetch: Download changes but don't merge
git fetch origin

# Pull: Fetch + merge
git pull origin main
```

**Difference:**
- `fetch`: Safe, just downloads
- `pull`: Modifies your working directory

### Pull Strategies

```bash
# Pull with merge (default)
git pull origin main

# Pull with rebase (cleaner history)
git pull --rebase origin main

# Pull with fast-forward only (safest)
git pull --ff-only origin main
```

**When to use each:**
- **Merge**: Team projects, preserves all history
- **Rebase**: Personal projects, cleaner linear history
- **Fast-forward only**: When you haven't made local changes

### Handle Unrelated Histories

```bash
# When local and remote have no common ancestor
git pull origin main --allow-unrelated-histories

# Common scenario: Created repo on GitHub with README,
# initialized locally separately
```

---

## Troubleshooting & Debugging

### Common Debugging Commands

```bash
# 1. Check current status
git status

# 2. See what changed
git diff

# 3. See staged changes
git diff --staged

# 4. Check commit history
git log --oneline

# 5. See remote info
git remote -v

# 6. Check current branch
git branch

# 7. See all branches (including remote)
git branch -a

# 8. Verify git config
git config --list
```

### Diagnose Push Issues

```bash
# Step 1: Check remote URL
git remote -v
# Ensure URL is correct (no typos like .gi instead of .git)

# Step 2: Test SSH
ssh -T git@github.com
# Should say "successfully authenticated"

# Step 3: Check branch
git branch
# Ensure you're on the right branch

# Step 4: Check if upstream is set
git branch -vv
# Shows upstream tracking

# Step 5: Fetch to see remote state
git fetch origin
git log origin/main --oneline
# See what's on remote
```

### Diagnose Merge Conflicts

```bash
# Step 1: See conflicted files
git status
# Files under "Unmerged paths"

# Step 2: See conflict markers in file
cat <filename>
# Look for <<<<<<< HEAD, =======, >>>>>>> markers

# Step 3: View difference
git diff

# Step 4: After resolving conflicts
git add <resolved-files>
git commit  # or git rebase --continue
```

---

## Common Git Issues & Solutions

### Issue 1: "remote origin already exists"

**Error:**
```
error: remote origin already exists.
```

**Solution:**
```bash
# Option A: Remove and re-add
git remote remove origin
git remote add origin <correct-url>

# Option B: Change URL
git remote set-url origin <correct-url>
```

---

### Issue 2: "Updates were rejected" (non-fast-forward)

**Error:**
```
! [rejected]        main -> main (fetch first)
error: failed to push some refs
```

**Cause:** Remote has commits you don't have locally

**Solution:**
```bash
# Option A: Pull then push (creates merge commit)
git pull origin main
git push origin main

# Option B: Rebase then push (cleaner history)
git pull --rebase origin main
git push origin main

# Option C: Force push (ONLY if you're sure)
git push --force origin main  # DANGEROUS
```

---

### Issue 3: Merge Conflicts

**Error:**
```
CONFLICT (content): Merge conflict in README.md
Automatic merge failed; fix conflicts and then commit
```

**Solution:**
```bash
# Step 1: See conflicted files
git status

# Step 2: Open file and resolve conflicts
# Look for:
# <<<<<<< HEAD
# your changes
# =======
# their changes
# >>>>>>> commit-hash

# Step 3: Choose resolution
# Option A: Keep yours
git checkout --ours <filename>

# Option B: Keep theirs
git checkout --theirs <filename>

# Option C: Manually edit file

# Step 4: Mark as resolved
git add <filename>

# Step 5: Continue
git commit  # for merge
git rebase --continue  # for rebase
```

---

### Issue 4: Divergent Branches

**Error:**
```
hint: You have divergent branches and need to specify how to reconcile them.
```

**Solution:**
```bash
# Configure default behavior
git config pull.rebase false  # merge (default)
git config pull.rebase true   # rebase
git config pull.ff only       # fast-forward only

# Or specify per pull
git pull --rebase origin main
```

---

### Issue 5: Accidentally Committed to Wrong Branch

**Solution:**
```bash
# If commit not yet pushed
# Step 1: Move commit to new branch
git branch new-feature-branch

# Step 2: Reset current branch
git reset --hard HEAD~1

# Step 3: Switch to new branch
git checkout new-feature-branch
```

---

### Issue 6: Need to Undo Last Commit

**Solutions:**
```bash
# Undo commit, keep changes staged
git reset --soft HEAD~1

# Undo commit, keep changes unstaged
git reset --mixed HEAD~1  # or just: git reset HEAD~1

# Undo commit, discard changes
git reset --hard HEAD~1  # DANGEROUS: loses work

# Undo commit but create new commit (safe for pushed commits)
git revert HEAD
```

---

## Best Practices

### Commit Best Practices

1. **Commit often, push strategically**
   - Small, logical commits
   - Push when feature is complete

2. **Write good commit messages**
   ```bash
   # Good
   git commit -m "Add visual embedding service with Nomic client"

   # Bad
   git commit -m "updates"
   git commit -m "fix stuff"
   git commit -m "WIP"  # Work in progress (never push)
   ```

3. **Never commit secrets**
   ```bash
   # Create .gitignore FIRST
   echo ".env" >> .gitignore
   echo "*.key" >> .gitignore
   echo "secrets/" >> .gitignore

   git add .gitignore
   git commit -m "Add gitignore for secrets"
   ```

4. **Review before committing**
   ```bash
   git diff          # See changes
   git status        # See what's staged
   git add -p        # Interactive staging
   ```

### Branch Best Practices

1. **Never work directly on main**
   ```bash
   # Create feature branch
   git checkout -b feature/visual-embeddings

   # Work on feature
   git add .
   git commit -m "Implement visual embedding extraction"

   # Push feature branch
   git push -u origin feature/visual-embeddings

   # Later: Merge via Pull Request on GitHub
   ```

2. **Keep main clean**
   - main = production-ready code
   - Use branches for development
   - Merge via Pull Requests

3. **Branch naming conventions**
   ```bash
   feature/visual-embeddings
   bugfix/fix-opensearch-timeout
   hotfix/security-patch
   refactor/improve-caching
   docs/update-architecture
   ```

### Remote Best Practices

1. **Use meaningful remote names**
   ```bash
   git remote add origin git@github.com:sushiva/project.git
   git remote add upstream git@github.com:original/project.git
   git remote add sushiva git@github.com:sushiva/project.git
   git remote add sudhirshivaram git@github.com:sudhirshivaram/project.git
   ```

2. **Fetch regularly**
   ```bash
   # See what's changed on remote
   git fetch origin
   git log origin/main --oneline
   ```

---

## Quick Reference

### Essential Commands

```bash
# Setup
git init                          # Initialize repository
git config user.name "Name"       # Set name
git config user.email "email"     # Set email

# Basic workflow
git status                        # Check status
git add <file>                    # Stage file
git add .                         # Stage all
git commit -m "message"           # Commit
git push origin main              # Push to remote

# Remotes
git remote -v                     # List remotes
git remote add <name> <url>       # Add remote
git remote remove <name>          # Remove remote

# Branching
git branch                        # List branches
git branch <name>                 # Create branch
git checkout <name>               # Switch branch
git checkout -b <name>            # Create and switch

# Syncing
git fetch origin                  # Download changes
git pull origin main              # Fetch and merge
git push origin main              # Upload changes

# Debugging
ssh -T git@github.com            # Test SSH
git log --oneline                # View history
git diff                         # See changes
git remote -v                    # Check remotes
```

### Common Workflows

#### Start New Project
```bash
mkdir project && cd project
git init
git branch -m main
echo "# Project Name" > README.md
git add .
git commit -m "Initial commit"
git remote add origin <url>
git push -u origin main
```

#### Fix Typo in Remote URL
```bash
git remote -v                    # Check current URL
git remote remove origin         # Remove incorrect
git remote add origin <correct>  # Add correct
git push -u origin main          # Push
```

#### Resolve Merge Conflict
```bash
git pull origin main             # Triggers conflict
git status                       # See conflicted files
# Edit files, remove conflict markers
git add <resolved-files>
git commit -m "Resolve merge conflict"
git push origin main
```

#### Undo Last Commit (Not Pushed)
```bash
git reset --soft HEAD~1          # Undo, keep changes staged
# Fix changes
git commit -m "Corrected commit"
```

---

## Shell Utilities

### Navigation
```bash
pwd                              # Print working directory
cd <path>                        # Change directory
cd ~                            # Go to home
cd -                            # Go to previous directory
ls                              # List files
ls -lah                         # List with details, hidden files
tree                            # Directory tree view
```

### File Operations
```bash
mkdir -p <path>                 # Create directory (with parents)
touch <file>                    # Create empty file
cp <src> <dest>                 # Copy file
mv <src> <dest>                 # Move/rename file
rm <file>                       # Remove file
rm -rf <dir>                    # Remove directory (DANGEROUS)
cat <file>                      # View file contents
head -n 10 <file>              # First 10 lines
tail -n 10 <file>              # Last 10 lines
wc -l <file>                   # Count lines
```

### Debugging
```bash
which git                       # Find git executable
git --version                   # Check git version
echo $PATH                      # See PATH variable
env | grep GIT                  # See Git environment vars
```

---

## Troubleshooting Checklist

When Git command fails, check in order:

1. ✅ **Is Git installed?**
   ```bash
   git --version
   ```

2. ✅ **Am I in a Git repository?**
   ```bash
   git status
   # Should not say: "not a git repository"
   ```

3. ✅ **Is my config correct?**
   ```bash
   git config --list
   ```

4. ✅ **Are remotes set up correctly?**
   ```bash
   git remote -v
   ```

5. ✅ **Can I authenticate to GitHub?**
   ```bash
   ssh -T git@github.com
   ```

6. ✅ **Am I on the right branch?**
   ```bash
   git branch
   ```

7. ✅ **Are there uncommitted changes?**
   ```bash
   git status
   ```

8. ✅ **Is my working directory clean?**
   ```bash
   git status
   # Should say: "nothing to commit, working tree clean"
   ```

---

## Advanced Topics (For Future Reference)

### Git Hooks
```bash
# Pre-commit hook (runs before commit)
.git/hooks/pre-commit

# Example: Run tests before commit
#!/bin/sh
pytest tests/
```

### Git Aliases
```bash
# Create shortcuts
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.unstage 'reset HEAD --'

# Now you can use:
git st
git co main
git ci -m "message"
```

### Stash (Temporary Storage)
```bash
# Save work without committing
git stash

# List stashes
git stash list

# Apply most recent stash
git stash pop

# Apply specific stash
git stash apply stash@{0}
```

### Cherry-pick (Copy Commit)
```bash
# Copy specific commit to current branch
git cherry-pick <commit-hash>
```

### Interactive Rebase
```bash
# Edit last 3 commits
git rebase -i HEAD~3

# Options in editor:
# pick = keep commit
# reword = change commit message
# squash = combine with previous commit
# drop = remove commit
```

---

## Resources

### Official Documentation
- Git: https://git-scm.com/doc
- GitHub Docs: https://docs.github.com

### Learning Resources
- Pro Git Book (free): https://git-scm.com/book
- GitHub Learning Lab: https://lab.github.com
- Git Cheat Sheet: https://training.github.com/downloads/github-git-cheat-sheet/

### Tools
- GitKraken: GUI for Git
- SourceTree: Another Git GUI
- GitHub Desktop: Simple Git UI

---

**Last Updated:** December 2024
**Version:** 1.0
**Maintainer:** Sudhir Shivaram

---

## Notes

- All commands tested on Linux (Ubuntu/Debian)
- Most commands work on macOS and Git Bash (Windows)
- Always commit before trying risky operations
- When in doubt, create a backup branch: `git branch backup`
