from flask_script import Manager, Server
from flask_migrate import MigrateCommand, Migrate

from NBC import db, create_app
from NBC.views.models import mahasiswa
from NBC.views.models import nilai
from NBC.views.models import alumni
from NBC.views.models import training
from NBC.views.models import testing
from NBC.views.models import hasil
from NBC.views.models import user

app = create_app('dev')
manager = Manager(app)
migrate = Migrate(app, db)
server = Server(threaded=True, port=5000)

manager.add_command('runserver', server)
manager.add_command('db', MigrateCommand)

if __name__ == '__main__':
    manager.run()
