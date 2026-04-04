# Vexoo AI Platform - Production Deployment Guide

Deploying a multi-component AI platform requires separating the frontend and backend architectures to optimize for scalability and cost. 

For the **Vexoo AI Platform**, the expert path is to deploy the **Frontend statically to Vercel** (for the global CDN edge-caching) and the **FastAPI Backend dynamically to Render or Railway** (using a Docker container to ensure system dependencies, like Python data structures or ML libraries, isolate cleanly). 

---

## 1. Deploying the FastAPI Backend (Render / Railway)

I have already scaffolded the `Dockerfile` and `.dockerignore` in your project root to handle the complex module dependencies in this codebase.

### Preparing the backend for Production
1. Push your entire repository to GitHub. 
2. Create an account on [Render.com](https://render.com/) or [Railway.app](https://railway.app/). 

### Step-by-Step Render Deployment:
1. In your Render Dashboard, click **New +** and select **Web Service**.
2. Connect your GitHub repository.
3. Configure the application:
   - **Name:** `vexoo-api` (or similar)
   - **Region:** Choose the region closest to where your users will primarily be.
   - **Branch:** `main`
   - **Runtime:** `Docker` *(Render will automatically detect the `Dockerfile` we just created)*.
   - **Build Command / Start Command:** Leave blank or default, as Docker configures everything.
   - **Instance Type:** "Free" or "Starter" (AI libraries sometimes eat RAM, so a Starter $7/mo instance with 512MB RAM might be required if indexing very large documents).
4. Click **Create Web Service**. 
5. Render will now build your container and expose a URL (e.g., `https://vexoo-api-xyz.onrender.com`). Make note of this URL. 

---

## 2. Deploying the Next.js Frontend (Vercel)

Vercel provides a seamless CI/CD pipeline natively integrated with Next.js. 

### Preparing the frontend for Production
The frontend queries `http://localhost:8000` via the `.env.local` file setting `NEXT_PUBLIC_API_URL`. We need to dynamically map this on the production server.

### Step-by-Step Vercel Deployment:
1. Create an account on [Vercel](https://vercel.com).
2. Click **Add New... > Project** and link your GitHub repository.
3. **Crucial Configuration Point:** 
   Because your Next.js app is located in a sub-folder (`frontend/`), you **must** configure the **Root Directory**.
   - Under *Project Settings* in the import flow, click `Edit` next to **Root Directory** and select `frontend`.
4. Under **Environment Variables**, add the mapping for our backend:
   - **Name:** `NEXT_PUBLIC_API_URL`
   - **Value:** Insert the URL of your deployed Render backend (e.g., `https://vexoo-api-xyz.onrender.com`).
5. Click **Deploy**. Vercel will install dependencies, build the Next.js application, and generate a live URL.

---

## 3. Post-Deployment Optimization Checklist

- **CORS Mitigation:** Ensure `backend/main.py` has the CORS middleware set up correctly to accept requests from your new Vercel domain. By default, developers use `allow_origins=["*"]`, but tightening it to `["https://your-vercel-app-url.vercel.app"]` is the expert practice.
- **Cold Starts:** If you are using Render's free tier, note that the backend will spin down after inactivity. The first API request might take ~30 seconds as the Docker container boots up. Consider a cron-job to ping the health endpoint (`/health`) every 10 minutes to prevent cold starts.
- **Environment Parity**: Never commit `.env` or `.env.local` to git. Ensure any keys needed are updated actively in Render and Vercel dashbords.
