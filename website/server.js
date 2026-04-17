require("dotenv").config();
const express = require("express");
const path = require("path");
const { Pool } = require("pg");
const bcrypt = require("bcrypt");
const cors = require("cors");

const app = express();
const PORT = 3000;

// middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname)));

// serve frontend files
app.use(express.static(path.join(__dirname, "public")));

// postgres connection, add these variables to environment file
const pool = new Pool({
  user: process.env.DB_USER,
  host: process.env.DB_HOST,
  database: "magdan",
  password: process.env.POSTGRES_PASSWORD,
  port: 5432,
});

// routes
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

// create account
app.post("/signup", async (req, res) => {
  try {
    const { username, email, password } = req.body;

    if (!username || !email || !password) {
      return res.status(400).json({ message: "All fields are required." });
    }

    const existingUser = await pool.query(
      "SELECT * FROM users WHERE username = $1 OR email = $2",
      [username, email]
    );

    if (existingUser.rows.length > 0) {
      return res.status(409).json({ message: "Username or email already exists." });
    }

    const passwordHash = await bcrypt.hash(password, 10);

    await pool.query(
      "INSERT INTO users (username, email, password_hash) VALUES ($1, $2, $3)",
      [username, email, passwordHash]
    );

    res.status(201).json({ message: "Account created successfully." });
  } catch (error) {
    console.error("Signup error:", error);
    res.status(500).json({ message: "Server error." });
  }
});

// login
app.post("/login", async (req, res) => {
  try {
    const { username, password } = req.body;

    const result = await pool.query(
      "SELECT * FROM users WHERE username = $1",
      [username]
    );

    if (result.rows.length === 0) {
      return res.status(401).json({ message: "Invalid username or password." });
    }

    const user = result.rows[0];
    const passwordMatch = await bcrypt.compare(password, user.password_hash);

    if (!passwordMatch) {
      return res.status(401).json({ message: "Invalid username or password." });
    }

    res.status(200).json({
      message: "Login successful.",
      user: {
        id: user.id,
        username: user.username,
        email: user.email,
      },
    });
  } catch (error) {
    console.error("Login error:", error);
    res.status(500).json({ message: "Server error." });
  }
});

// =========================
// GET POSTS FROM DATABASE
// =========================
app.get("/posts", async (req, res) => {
  try {
    const result = await pool.query(`
      SELECT
        p.post_id,
        p."user_key" AS user_id,
        p.image_key,
        p.description_key,

        d."text" AS description_text,
        d.is_safe_content,

        u."username",
        u.num_of_posts,
        u.num_of_violations,

        i.image_path,
        i.label

      FROM dim_post AS p
      INNER JOIN dim_user AS u
        ON p."user_key" = u."user_id"
      INNER JOIN dim_description AS d
        ON p.description_key = d.description_id
      INNER JOIN dim_image i
        ON p.image_key= i.image_id
      ORDER BY p.post_id DESC;
    `);

    const posts = result.rows.map((row) => ({
      id: row.post_id,
      status: row.status,
      user_id: row.user_id,
      image: row.image,
      image_path: row.image_path,
      label: row.label,
      description_key: row.description_key,
      description_text: row.description_text,
      is_safe_content: row.is_safe_content,
      username: row.username,
      num_of_posts: row.num_of_posts,
      num_of_violations: row.num_of_violations,
    }));

    res.json({ posts });
  } catch (error) {
    console.error("Error fetching posts:", error);
    res.status(500).json({ message: "Failed to fetch posts" });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});