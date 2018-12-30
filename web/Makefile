SHELL := bash
export PATH := node_modules/.bin/:$(PATH)

## Variables ##
srcpath = .

## Rules ##

install-current:
	cd $(srcpath) && npm install && touch node_modules

node_modules: package.json
	npm install && touch node_modules

install: node_modules .env install-current

compile: install
	tsc --project $(srcpath)/tsconfig.json

run: compile
	node $(srcpath)/index.js

dev: compile
	node $(srcpath)/index.js | pino-pretty

infra:
	docker-compose up -d --force-recreate

infra-stop:
	docker-compose stop

infra-restart: infra-stop infra

watch:
	nodemon --watch src --ext ts --exec "make run"

lint:
	tslint --project tsconfig.json --project --fix -r ./tslint.json

lint-error:
	tslint --project tsconfig.json --project

clean:
	find $(srcpath) -name '*.js' -delete -o -name '*.js.map' -delete

clean-all:
	$(MAKE) clean > /dev/null
	find . -name "node_modules" -type d -exec rm -r "{}" \;

all: run

rules :=	all				    \
			install-current  \
			compile				\
			run					\
			run-watch			\
			install				\
			lint				\
			lint-error		    \
			clean		        \
			clean-all		    \
			infra						\
			infra-stop			\
			infra-restart

.PHONY: $(rules)
.SILENT: $(rules) node_modules

# Local makefile to modify flags
-include local.mk
