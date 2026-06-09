# Publishing to GitHub

This guide covers everything from creating an account to having a live, public
repository that others can clone and use.

---

## Part 1 — Create a GitHub account

1. Go to **https://github.com** and click **Sign up**.
2. Enter your email address, create a password, and choose a username.
   - Your username appears in every repository URL:
     `github.com/brakuta/Date-Palm-Tree-Segmentation-from-WorldView3-Satellite-Data`
   - Use something professional — most researchers use their name or an
     institutional abbreviation (e.g. `mbgibril`).
3. Complete the email verification step GitHub sends you.
4. On the "What are you interested in?" screen you can skip or fill it in;
   it does not affect anything.

---

## Part 2 — Install Git on your machine

Git is the version-control program that sends your code to GitHub.

**Windows:**
1. Download the installer from **https://git-scm.com/download/win** and run it.
   Accept all defaults.
2. After installation, open **Git Bash** (search for it in the Start menu).
   All commands below are run in Git Bash.

**macOS:**
```bash
# Git ships with Xcode Command Line Tools; install them if missing:
xcode-select --install
```

**Linux (Ubuntu / Debian):**
```bash
sudo apt update && sudo apt install git -y
```

Verify:
```bash
git --version    # should print e.g. git version 2.43.0
```

---

## Part 3 — Configure Git with your identity

Git stamps every commit with your name and email. Do this once:

```bash
git config --global user.name  "Mohamed Barakat A. Gibril"
git config --global user.email "your.email@institution.ac.ae"
```

---

## Part 4 — Repository on GitHub

Your repository is already created at:
https://github.com/brakuta/Date-Palm-Tree-Segmentation-from-WorldView3-Satellite-Data

Skip to Part 5.

---

## Part 5 — Connect your local folder to GitHub and push

Open a terminal (Git Bash on Windows, Terminal on macOS/Linux) and navigate
to the repository folder:

```bash
cd path/to/Date-Palm-Tree-Segmentation-from-WorldView3-Satellite-Data
```

Run these commands **in order**:

```bash
# 1. Initialise Git tracking in the folder (only needed once)
git init

# 2. Stage every file for the first commit
git add .

# 3. Review what will be committed (optional but recommended)
git status

# 4. Create the first commit
git commit -m "Initial release: date palm segmentation toolkit"

# 5. Name the main branch 'main' (GitHub's default)
git branch -M main

# 6. Link the local folder to the GitHub repository you just created
#    Replace brakuta with your actual GitHub username
git remote add origin https://github.com/brakuta/Date-Palm-Tree-Segmentation-from-WorldView3-Satellite-Data.git

# 7. Push (upload) the code to GitHub
git push -u origin main
```

GitHub will ask for your username and password at step 7. **Your GitHub
password does not work here.** You need a Personal Access Token:

1. Go to **https://github.com/settings/tokens** → **Generate new token
   (classic)**.
2. Give it a note (e.g. "push from laptop"), set expiration to 90 days or
   No expiration, and check the **repo** scope.
3. Click **Generate token** and **copy it immediately** (it is shown only once).
4. Paste the token as the password when Git prompts you.

After the push, refresh the GitHub repository page — your code is now live.

---

## Part 6 — Confirm large files are excluded

Model weights (`.pth`) and rasters (`.tif`, `.gpkg`) must not be committed —
GitHub rejects files over 100 MB. The `.gitignore` already excludes them, but
verify before every push:

```bash
git ls-files | grep -E "\.(pth|tif|gpkg)$"   # should return nothing
```

If a large file appears, remove it from tracking without deleting it locally:

```bash
git rm --cached weights/segformer_b5_ms.pth
echo "weights/" >> .gitignore
git commit -m "Remove large files from tracking"
```

---

## Part 7 — Fill in placeholders

Before pushing, update the three placeholders in the repository:

```bash
# 1. Set your GitHub username in the docs
#    (find all occurrences and replace)
grep -rn "brakuta" docs/ README.md
# Open each file and replace brakuta with your actual username.
```

```python
# 2. Set your Hugging Face repo in palmseg/weights_manifest.py
HF_REPO_ID = 'brakuta/date-palm-wv3-models'   # your HF username / repo name
```

```yaml
# 3. Fill in CITATION.cff
orcid: "https://orcid.org/0000-0000-0000-0000"  # your ORCID (orcid.org)
affiliation: "Research Institute of Sciences and Engineering, University of Sharjah"
repository-code: "https://github.com/brakuta/Date-Palm-Tree-Segmentation-from-WorldView3-Satellite-Data"
```

Commit these changes:

```bash
git add .
git commit -m "Set author metadata and repository links"
git push
```

---

## Part 8 — Make the repository visible and citable

**Add topics** so people can find it:

1. On the GitHub repository page, click the gear icon next to **About**.
2. Add topics: `remote-sensing`, `semantic-segmentation`, `date-palm`,
   `worldview-3`, `mmsegmentation`, `pytorch`, `tree-mapping`.

**Enable the citation button:**

The `CITATION.cff` file you already have causes GitHub to show a
**"Cite this repository"** button automatically on the right-hand panel.

**Create a release:**

```bash
git tag v0.1.0
git push --tags
```

Then on GitHub: **Releases → Create a new release** → select tag `v0.1.0`
→ fill the description → publish.

---

## Part 9 — Updating the repository later

Every time you change the code or documentation:

```bash
git add .
git commit -m "Describe what changed"
git push
```

That is all that is required for routine updates.

---

## Checklist before going public

- [ ] `git ls-files | grep -E "\.(pth|tif|gpkg)$"` returns nothing
- [ ] `brakuta` replaced in all docs and README
- [ ] `HF_REPO_ID` set in `palmseg/weights_manifest.py`
- [ ] `CITATION.cff` has your ORCID, affiliation, and repository URL
- [ ] Repository topics added on GitHub
- [x] Repository created: https://github.com/brakuta/Date-Palm-Tree-Segmentation-from-WorldView3-Satellite-Data
- [ ] Weights uploaded to Hugging Face (see `docs/HUGGINGFACE_WEIGHTS.md`)
