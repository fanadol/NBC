from flask_script import Manager, Server
from flask_migrate import MigrateCommand, Migrate

from NBC import db, create_app

app = create_app('dev')
manager = Manager(app)
migrate = Migrate(app, db)
server = Server(threaded=True, port=5000)

manager.add_command('runserver', server)
manager.add_command('db', MigrateCommand)

if __name__ == '__main__':
    manager.run()
