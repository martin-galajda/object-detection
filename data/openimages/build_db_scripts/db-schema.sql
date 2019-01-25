-- A table that stores a images
create table IF NOT EXISTS images(
  id varchar primary key not null,
  url varchar not null,
  data BLOB,
  image_data BLOB,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- A table that stores a image labels
create table IF NOT EXISTS image_labels(
  -- mid
  id varchar primary key not null,
  image_id varchar references images(id) not null,
  label_name varchar not null
);

CREATE TRIGGER update_updated_at UPDATE OF id, url, data ON images
BEGIN
  UPDATE images SET updated_at=CURRENT_TIMESTAMP WHERE id=id;
END;
